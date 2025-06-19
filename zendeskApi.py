import requests
import json
import base64
import os
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
import html

class ZendeskDocFetcher:
    def __init__(self):
        # Configuration
        self.zendesk_subdomain = "trilogyeffective"
        self.zendesk_user = "TIESConnectHelpCenterBot@trilogyes.com"
        self.encoded_token = os.environ.get('ZENDESK_API_TOKEN')
        
        # Decode the token if it's base64 encoded
        if self.encoded_token.endswith('=='):
            try:
                self.api_token = base64.b64decode(self.encoded_token).decode('utf-8')
            except Exception as e:
                print(f"Error decoding token: {e}")
                self.api_token = self.encoded_token
        else:
            self.api_token = self.encoded_token
        
        # Set up authentication
        self.auth = (f"{self.zendesk_user}/token", self.api_token)
        
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"zendesk_docs_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track downloaded images to avoid duplicates
        self.downloaded_images = set()
        
        # Rate limiting
        self.request_count = 0
        self.max_requests_per_minute = 200  # Zendesk rate limit
        self.start_time = time.time()

    def check_rate_limit(self):
        """Ensure we don't exceed Zendesk's rate limit"""
        self.request_count += 1
        if self.request_count >= self.max_requests_per_minute:
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                sleep_time = 60 - elapsed + 1  # Add 1 second buffer
                print(f"Rate limit approaching. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.request_count = 0
            self.start_time = time.time()

    def fetch_categories(self):
        """Fetch all categories from Zendesk"""
        self.check_rate_limit()
        url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/help_center/en-us/categories.json"
        
        print(f"Fetching categories from {url}...")
        
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        
        data = response.json()
        categories = data.get('categories', [])
        
        # Find version 25.0 category
        version_category = None
        for category in categories:
            if category['name'] == '25.0' or category['name'].startswith('25.0 '):
                version_category = category
                break
        
        if not version_category:
            print("Version 25.0 category not found. Available categories:")
            for cat in categories:
                print(f"- {cat['name']} (ID: {cat['id']})")
            return None
        
        print(f"Found version 25.0 category: {version_category['name']} (ID: {version_category['id']})")
        
        # Save all categories for reference
        with open(f"{self.output_dir}/all_categories.json", 'w', encoding='utf-8') as f:
            json.dump(data, indent=2, fp=f)
        
        return version_category

    def fetch_sections(self, category_id):
        """Fetch all sections in a category"""
        self.check_rate_limit()
        url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/help_center/en-us/categories/{category_id}/sections.json"
        
        print(f"Fetching sections for category ID {category_id}...")
        
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        
        data = response.json()
        sections = data.get('sections', [])
        
        print(f"Found {len(sections)} sections")
        
        # Save sections
        with open(f"{self.output_dir}/sections_category_{category_id}.json", 'w', encoding='utf-8') as f:
            json.dump(data, indent=2, fp=f)
        
        return sections

    def fetch_articles(self, section_id):
        """Fetch all articles in a section"""
        self.check_rate_limit()
        url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/help_center/en-us/sections/{section_id}/articles.json"
        
        print(f"Fetching articles for section ID {section_id}...")
        
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get('articles', [])
        
        # Check for pagination
        while data.get('next_page'):
            self.check_rate_limit()
            response = requests.get(data['next_page'], auth=self.auth)
            response.raise_for_status()
            data = response.json()
            articles.extend(data.get('articles', []))
        
        print(f"Found {len(articles)} articles")
        
        # Save articles
        with open(f"{self.output_dir}/articles_section_{section_id}.json", 'w', encoding='utf-8') as f:
            json.dump({"articles": articles}, indent=2, fp=f)
        
        return articles

    def fetch_article_content(self, article_id):
        """Fetch content for a specific article"""
        self.check_rate_limit()
        url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/help_center/articles/{article_id}.json"
        
        print(f"Fetching content for article ID {article_id}...")
        
        try:
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            
            data = response.json()
            article = data.get('article', {})
            
            # Save article data as JSON
            articles_dir = f"{self.output_dir}/articles_json"
            os.makedirs(articles_dir, exist_ok=True)
            
            with open(f"{articles_dir}/article_{article_id}.json", 'w', encoding='utf-8') as f:
                json.dump(article, indent=2, fp=f)
            
            # No longer generating HTML files
            print(f"Saved article {article_id} as JSON (skipping HTML generation)")
            
            return article
        except Exception as e:
            print(f"Error fetching article {article_id}: {e}")
            return {}

    def extract_images_from_html(self, html_content, article_id):
        """Extract image URLs from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        images = []
        
        # Find all img tags
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                # Clean up the URL
                src = html.unescape(src)
                images.append(src)
        
        print(f"Found {len(images)} images in article {article_id} (not downloading)")
        return images

    def download_image(self, image_url, article_id):
        """Log image URL but don't download"""
        # Skip if already logged
        if image_url in self.downloaded_images:
            print(f"Image already logged: {image_url}")
            return None
        
        # Extract filename from URL
        filename = os.path.basename(image_url.split('?')[0])
        if not filename:
            filename = f"image_{len(self.downloaded_images) + 1}.png"
        
        # Create images directory for metadata only
        images_dir = f"{self.output_dir}/images_metadata"
        os.makedirs(images_dir, exist_ok=True)
        
        # Log the image metadata without downloading
        print(f"Logging image (not downloading): {image_url}")
        
        # Mark as processed
        self.downloaded_images.add(image_url)
        
        # Save image metadata
        with open(f"{images_dir}/{filename}.json", 'w', encoding='utf-8') as f:
            json.dump({
                "url": image_url,
                "article_id": article_id,
            }, indent=2, fp=f)
        
        return None

    def fetch_attachments(self, article_id):
        """Fetch metadata for all attachments for an article without downloading"""
        self.check_rate_limit()
        url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/help_center/articles/{article_id}/attachments.json"
        
        print(f"Fetching attachment metadata for article ID {article_id}...")
        
        try:
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            
            data = response.json()
            attachments = data.get('article_attachments', [])
            
            # Save attachments metadata
            attachments_dir = f"{self.output_dir}/attachments_metadata"
            os.makedirs(attachments_dir, exist_ok=True)
            
            with open(f"{attachments_dir}/attachments_article_{article_id}.json", 'w', encoding='utf-8') as f:
                json.dump(data, indent=2, fp=f)
            
            # Log each attachment without downloading
            for attachment in attachments:
                content_url = attachment.get('content_url')
                if content_url:
                    self.log_attachment(content_url, attachment, article_id)
            
            return attachments
        except Exception as e:
            print(f"Error fetching attachments for article {article_id}: {e}")
            return []

    def log_attachment(self, content_url, attachment_data, article_id):
        """Log attachment metadata without downloading"""
        # Skip if already logged
        if content_url in self.downloaded_images:
            print(f"Attachment already logged: {content_url}")
            return None
        
        # Extract filename from attachment data or URL
        filename = attachment_data.get('file_name') or os.path.basename(content_url.split('?')[0])
        if not filename:
            filename = f"attachment_{attachment_data.get('id')}"
        
        print(f"Logging attachment (not downloading): {filename}")
        
        # Mark as processed
        self.downloaded_images.add(content_url)
        
        return None

    def create_documentation_index(self, category, sections, articles_by_section):
        """Create a JSON index of the documentation structure"""
        index = {
            "category": category,
            "sections": sections,
            "articles_by_section": articles_by_section
        }
        
        # Save index as JSON
        with open(f"{self.output_dir}/documentation_index.json", 'w', encoding='utf-8') as f:
            json.dump(index, indent=2, fp=f)
        
        print(f"Created documentation index JSON")
        
        # Skip HTML index generation

    def fetch_all_documentation(self):
        """Fetch all documentation for version 25.0"""
        try:
            # Step 1: Fetch categories and find version 25.0
            category = self.fetch_categories()
            if not category:
                return False
            
            # Step 2: Fetch all sections in the category
            sections = self.fetch_sections(category['id'])
            
            # Step 3: Fetch all articles in each section
            articles_by_section = {}
            all_articles = []
            
            for section in sections:
                articles = self.fetch_articles(section['id'])
                articles_by_section[section['id']] = articles
                all_articles.extend(articles)
            
            print(f"Total articles found: {len(all_articles)}")
            
            # Step 4: Fetch content for each article
            for article in all_articles:
                article_data = self.fetch_article_content(article['id'])
                
                # Step 5: Extract and log images from article content (without downloading)
                if article_data.get('body'):
                    image_urls = self.extract_images_from_html(article_data['body'], article['id'])
                    for image_url in image_urls:
                        self.download_image(image_url, article['id'])
                
                # Step 6: Fetch attachment metadata (without downloading)
                self.fetch_attachments(article['id'])
            
            # Step 7: Create documentation index (JSON only)
            self.create_documentation_index(category, sections, articles_by_section)
            
            print(f"\nDocumentation fetching complete!")
            print(f"All files saved to: {os.path.abspath(self.output_dir)}")
            print(f"JSON data saved to: {os.path.abspath(f'{self.output_dir}/articles_json')}")
            print(f"Note: HTML files were not generated, and images/attachments were not downloaded.")
            
            return True
        
        except Exception as e:
            print(f"Error fetching documentation: {e}")
            return False

if __name__ == "__main__":
    print("=== Zendesk Documentation Fetcher ===")
    print("This script will fetch all documentation for version 25.0")
    print("The documentation will be saved to a local directory")
    
    fetcher = ZendeskDocFetcher()
    fetcher.fetch_all_documentation()