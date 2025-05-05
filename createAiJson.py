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
            
            # Extract full content as plain text with image references
            full_content = []
            
            # Process all elements in order
            root_element = soup.body if soup.body else soup
            for element in root_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'div', 'img', 'span'], recursive=True):
                # Skip empty elements or those that are just containers with no direct text
                if element.name in ['div', 'span'] and not element.get_text(strip=True) and not element.find('img'):
                    continue
                
                # Handle headings
                if element.name.startswith('h'):
                    heading_level = int(element.name[1])
                    heading_text = element.get_text().strip()
                    
                    if heading_text:
                        # Add heading with appropriate formatting
                        heading_prefix = "#" * heading_level
                        full_content.append(f"{heading_prefix} {heading_text}")
                        full_content.append("")  # Add blank line after heading
                
                # Handle paragraphs and text content
                elif element.name == 'p' and element.get_text(strip=True):
                    text_content = element.get_text().strip()
                    full_content.append(text_content)
                    full_content.append("")  # Add blank line after paragraph
                
                # Handle list items
                elif element.name == 'li' and element.get_text(strip=True):
                    text_content = element.get_text().strip()
                    # Check if parent is ordered or unordered list
                    if element.parent.name == 'ol':
                        # Find position in ordered list
                        position = 1
                        for sibling in element.previous_siblings:
                            if sibling.name == 'li':
                                position += 1
                        full_content.append(f"{position}. {text_content}")
                    else:
                        full_content.append(f"* {text_content}")
                
                # Handle images
                elif element.name == 'img' or (element.name in ['span', 'div'] and element.find('img')):
                    # Get the image element
                    img = element if element.name == 'img' else element.find('img')
                    if not img:
                        continue
                        
                    src = img.get('src', '')
                    alt_text = img.get('alt', '').strip()
                    
                    # Extract image ID from URL
                    image_id_match = re.search(r'article_attachments/(\d+)', src)
                    if image_id_match:
                        image_id = image_id_match.group(1)
                        # Add image reference to full content
                        img_alt = f" '{alt_text}'" if alt_text else ""
                        full_content.append(f"![Image{img_alt}](IMAGE_ID:{image_id})")
                        full_content.append("")  # Add blank line after image
            
            # Join full content into a single string
            full_content_text = "\n".join(full_content)
            
            # Now create the structured document representation
            document_structure = []
            
            # Track section hierarchy
            section_stack = []
            current_section = None
            
            # Process all elements again for structured content
            for element in root_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'div', 'img', 'span'], recursive=True):
                # Skip empty elements or those that are just containers with no direct text
                if element.name in ['div', 'span'] and not element.get_text(strip=True) and not element.find('img'):
                    continue
                
                # Handle headings - start new sections
                if element.name.startswith('h'):
                    heading_level = int(element.name[1])
                    heading_text = element.get_text().strip()
                    
                    if heading_text:
                        # Pop sections from stack until we find a parent level
                        while section_stack and section_stack[-1]["level"] >= heading_level:
                            section_stack.pop()
                        
                        # Create a new section
                        new_section = {
                            "type": "section",
                            "heading": heading_text,
                            "level": heading_level,
                            "content": []
                        }
                        
                        # Add to parent section or document structure
                        if section_stack:
                            section_stack[-1]["content"].append(new_section)
                        else:
                            document_structure.append(new_section)
                        
                        # Update current section and stack
                        current_section = new_section
                        section_stack.append(new_section)
                
                # Handle paragraphs, lists, and other text content
                elif element.name in ['p', 'ul', 'ol'] and element.get_text(strip=True):
                    text_content = element.get_text().strip()
                    
                    # Handle lists specially to preserve structure
                    if element.name in ['ul', 'ol']:
                        list_items = []
                        for li in element.find_all('li', recursive=False):
                            list_items.append(li.get_text().strip())
                        
                        if list_items:
                            list_content = {
                                "type": "list",
                                "list_type": "bullet" if element.name == 'ul' else "numbered",
                                "items": list_items
                            }
                            
                            if current_section:
                                current_section["content"].append(list_content)
                            else:
                                # If no section yet, create a default one
                                current_section = {
                                    "type": "section",
                                    "heading": "Introduction",
                                    "level": 1,
                                    "content": [list_content]
                                }
                                document_structure.append(current_section)
                                section_stack = [current_section]
                    
                    # Handle regular paragraphs
                    elif text_content:
                        if current_section:
                            current_section["content"].append({
                                "type": "text",
                                "content": text_content
                            })
                        else:
                            # If no section yet, create a default one
                            current_section = {
                                "type": "section",
                                "heading": "Introduction",
                                "level": 1,
                                "content": [{
                                    "type": "text",
                                    "content": text_content
                                }]
                            }
                            document_structure.append(current_section)
                            section_stack = [current_section]
                
                # Handle images
                elif element.name == 'img' or (element.name in ['span', 'div'] and element.find('img')):
                    # Get the image element
                    img = element if element.name == 'img' else element.find('img')
                    if not img:
                        continue
                        
                    src = img.get('src', '')
                    alt_text = img.get('alt', '').strip()
                    
                    # Extract image ID from URL
                    image_id_match = re.search(r'article_attachments/(\d+)', src)
                    if image_id_match and current_section:
                        image_id = image_id_match.group(1)
                        
                        # Find context before and after
                        context_before = ""
                        context_after = ""
                        
                        # Look for previous paragraph or list item
                        prev_elem = img.find_previous(['p', 'li'])
                        if prev_elem:
                            context_before = prev_elem.get_text().strip()
                        
                        # Look for next paragraph or list item
                        next_elem = img.find_next(['p', 'li'])
                        if next_elem:
                            context_after = next_elem.get_text().strip()
                        
                        # If still no context, use section heading
                        if not context_before and current_section:
                            context_before = current_section["heading"]
                        
                        # Create image reference
                        image_ref = {
                            "type": "image",
                            "id": image_id,
                            "url": src,
                            "alt_text": alt_text,
                            "position": {
                                "section_heading": current_section["heading"],
                                "section_level": current_section["level"]
                            },
                            "context_before": context_before,
                            "context_after": context_after
                        }
                        
                        # Add image to current section
                        current_section["content"].append(image_ref)
            
            # Extract all images for separate reference
            image_attachments = []
            
            for img in soup.find_all('img'):
                src = img.get('src', '')
                
                # Extract image ID from URL
                image_id_match = re.search(r'article_attachments/(\d+)', src)
                if image_id_match:
                    image_id = image_id_match.group(1)
                    
                    # Find the nearest heading
                    nearest_heading = img.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    heading_text = nearest_heading.get_text().strip() if nearest_heading else "Introduction"
                    heading_level = int(nearest_heading.name[1]) if nearest_heading else 1
                    
                    # Find paragraphs before and after
                    paragraphs_before = []
                    paragraphs_after = []
                    
                    # Get up to 3 paragraphs before
                    current = img
                    for _ in range(3):
                        prev = current.find_previous(['p', 'li'])
                        if prev and prev.get_text().strip():
                            paragraphs_before.insert(0, prev.get_text().strip())
                        current = prev if prev else current
                    
                    # Get up to 3 paragraphs after
                    current = img
                    for _ in range(3):
                        next_elem = current.find_next(['p', 'li'])
                        if next_elem and next_elem.get_text().strip():
                            paragraphs_after.append(next_elem.get_text().strip())
                        current = next_elem if next_elem else current
                    
                    # Create detailed image metadata
                    image_attachment = {
                        "id": image_id,
                        "url": src,
                        "alt_text": img.get('alt', '').strip(),
                        "position": {
                            "section_heading": heading_text,
                            "section_level": heading_level
                        },
                        "context_before": "\n".join(paragraphs_before),
                        "context_after": "\n".join(paragraphs_after)
                    }
                    
                    image_attachments.append(image_attachment)
            
            # Create document with rich structure
            document = {
                "id": str(article_id),
                "title": title,
                "last_updated": last_updated,
                "url": url,
                "full_content": full_content_text,
                "document_structure": document_structure,
                "image_attachments": image_attachments,
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