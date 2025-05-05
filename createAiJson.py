import json
import os
import re
from bs4 import BeautifulSoup
from html import unescape

def process_zendesk_articles():
    # Directory containing the article JSON files
    articles_dir = "zendesk_docs_20250422_115533/articles_json/"
    
    # Output file for processed documents
    output_file = "processed_zendesk_docs_v2.json"
    
    documents = []
    
    # Process each article JSON file
    for filename in os.listdir(articles_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(articles_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                article_data = json.load(f)
            
            # Extract basic metadata
            article_id = article_data.get("id")
            title = article_data.get("title")
            last_updated = article_data.get("updated_at")
            url = article_data.get("html_url")
            labels = article_data.get("label_names", [])
            
            # Extract the HTML body content
            html_body = article_data.get("body", "")
            
            # Parse the HTML
            soup = BeautifulSoup(html_body, "html.parser")
            
            # Extract structured content
            structured_content = []
            
            # Process headings and their content
            current_section = None
            
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'div']):
                # Check if it's a heading
                if element.name.startswith('h'):
                    heading_level = int(element.name[1])
                    heading_text = element.get_text().strip()
                    
                    if heading_text:
                        # Create a new section
                        current_section = {
                            "type": "section",
                            "heading": heading_text,
                            "level": heading_level,
                            "content": []
                        }
                        structured_content.append(current_section)
                
                # If it's a paragraph, list, or other content element
                elif current_section is not None and element.name in ['p', 'ul', 'ol', 'div']:
                    # Extract text content
                    text = element.get_text().strip()
                    if text:
                        current_section["content"].append(text)
            
            # Extract image attachments with better context
            attachments = []
            for img in soup.find_all('img'):
                src = img.get('src', '')
                
                # Extract image ID from URL
                image_id_match = re.search(r'article_attachments/(\d+)', src)
                if image_id_match:
                    image_id = image_id_match.group(1)
                    
                    # Get better context before and after the image
                    # Find the image's position in the document
                    img_parent = img.parent
                    
                    # Get context before - look for previous paragraph or heading
                    context_before = ""
                    prev_elem = img_parent.find_previous(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                    if prev_elem:
                        context_before = prev_elem.get_text().strip()
                    
                    # Get context after - look for next paragraph or heading
                    context_after = ""
                    next_elem = img_parent.find_next(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                    if next_elem:
                        context_after = next_elem.get_text().strip()
                    
                    # If we couldn't find good context, try to get text from parent
                    if not context_before and not context_after:
                        parent_text = img_parent.get_text().strip()
                        if parent_text:
                            # Remove the alt text if it's in the parent text
                            alt_text = img.get('alt', '').strip()
                            if alt_text:
                                parent_text = parent_text.replace(alt_text, '').strip()
                            
                            if parent_text:
                                context_before = parent_text
                    
                    # Get the full HTML context for reference
                    full_context = str(img_parent)
                    
                    attachments.append({
                        "id": image_id,
                        "url": src,
                        "context_before": context_before,
                        "context_after": context_after,
                        "full_context": full_context,
                        "alt_text": img.get('alt', '').strip()
                    })
            
            # Create document
            document = {
                "id": str(article_id),
                "title": title,
                "last_updated": last_updated,
                "url": url,
                "structured_content": structured_content,
                "attachments": attachments,
                "references": [],
                "labels": labels
            }
            
            documents.append(document)
            print(f"Processed article: {title}")
    
    # Write processed documents to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2)
    
    print(f"Processed {len(documents)} documents and saved to {output_file}")

# Run the function
if __name__ == "__main__":
    process_zendesk_articles()