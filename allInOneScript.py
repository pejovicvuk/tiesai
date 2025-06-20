import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

class TIESArticleProcessor:
    """
    Processes TIES articles into optimized RAG chunks and uploads to Pinecone
    """
    
    def __init__(self):
        # Section ID to category mapping
        self.section_mappings = {
            "33996954647565": "new_in_release",
            "33997001085965": "learning_path", 
            "33997001109133": "foundations_and_concepts",
            "33996954739085": "how_to",
            "33996954772621": "reference",
            "33997001194381": "scheduling_app",
            "33996954814605": "customer_activity",
            "33996954831885": "fuel_calculator"
        }
        
        # Category-specific chunking strategies
        self.chunking_strategies = {
            "new_in_release": {"max_size": 1800, "split_pattern": r'\n(?=#{1,6}|[A-Z][^:]*:|\*\*[^*]+\*\*)'},
            "learning_path": {"max_size": 2500, "split_pattern": r'\n(?=#{1,6}|\d+\.|[A-Z][^:]*Learning|Overview)'},
            "foundations_and_concepts": {"max_size": 1800, "split_pattern": r'\n(?=#{1,6}|[A-Z][^:]*:|What|How)'},
            "how_to": {"max_size": 1200, "split_pattern": r'\n(?=#{1,6}|\d+\.|Step|\* )'},
            "reference": {"max_size": 800, "split_pattern": r'\n(?=#{1,6}|\*\*[^*]+\*\*|\* )'},
            "scheduling_app": {"max_size": 1800, "split_pattern": r'\n(?=#{1,6}|Overview|Process)'},
            "customer_activity": {"max_size": 2500, "split_pattern": r'\n(?=#{1,6}|Overview|Process|Step)'},
            "fuel_calculator": {"max_size": 1800, "split_pattern": r'\n(?=#{1,6}|Process|Configuration)'}
        }

        # Role keywords for detection
        self.role_keywords = {
            "trader": ["trader", "trading", "buy", "sell", "deal", "physical", "financial"],
            "scheduler": ["scheduler", "scheduling", "nomination", "balance", "move", "transport"],
            "accountant": ["accountant", "accounting", "invoice", "payment", "accrual", "ar", "ap"],
            "administrator": ["admin", "administrator", "configuration", "setup", "management"],
            "contract_administrator": ["contract", "agreement", "terms", "bilateral"],
            "all_users": ["user", "general", "overview", "foundation", "basic"],
            "shipper": ["shipper", "customer", "caw", "nominator", "hub"],
            "operator": ["operator", "facility", "pipeline", "plant"],
            "analyst": ["analyst", "report", "query", "analysis"],
            "risk_manager": ["risk", "manager", "margin", "profit"],
            "back_office": ["back_office", "confirmation", "settlement"]
        }

    def clean_html_content(self, html_content: str) -> str:
        """Extract clean text from HTML while preserving structure"""
        if not html_content:
            return ""
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and metadata elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()
        
        # Convert structural elements to readable text
        for br in soup.find_all("br"):
            br.replace_with("\n")
            
        # Add spacing after headers and paragraphs
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
            element.append("\n")
            
        # Extract text and clean up
        text = soup.get_text()
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)      # Clean multiple spaces
        text = text.strip()
        
        return text

    def determine_category(self, article: Dict[str, Any]) -> str:
        """Determine article category from section_id"""
        section_id = str(article.get("section_id", ""))
        return self.section_mappings.get(section_id, "reference")

    def extract_version_info(self, content: str, title: str) -> str:
        """Extract version information from content or title"""
        version_patterns = [
            r'[Vv]ersion\s*([\d.]+)',
            r'TIES\s*([\d.]+)',
            r'v([\d.]+)',
            r'(\d{4}\.\d{2})'  # Year.month format
        ]
        
        text_to_search = f"{title} {content}"
        
        for pattern in version_patterns:
            matches = re.findall(pattern, text_to_search)
            if matches:
                return matches[0]
                
        return "latest"

    def determine_user_roles(self, content: str, title: str, category: str) -> List[str]:
        """Determine target user roles from content, title, and category"""
        text = f"{title} {content}".lower()
        detected_roles = []
        
        for role, keywords in self.role_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_roles.append(role)
        
        # Remove duplicates and apply category defaults if none found
        detected_roles = list(set(detected_roles))
        
        if not detected_roles:
            defaults = {
                "new_in_release": ["all_users"],
                "learning_path": ["all_users"],
                "foundations_and_concepts": ["all_users"],
                "how_to": ["all_users"],
                "reference": ["administrator"],
                "scheduling_app": ["scheduler"],
                "customer_activity": ["shipper"],
                "fuel_calculator": ["operator"]
            }
            detected_roles = defaults.get(category, ["all_users"])
            
        return detected_roles

    def extract_keywords(self, title: str, content: str, category: str) -> List[str]:
        """Extract relevant keywords from title and content"""
        text = f"{title} {content}".lower()
        
        # TIES-specific keywords
        ties_keywords = [
            "ties", "connect", "trading", "scheduling", "nomination",
            "contract", "pipeline", "facility", "station", "deal",
            "accounting", "pricing", "fuel", "balance", "shipper",
            "operator", "meter", "volume", "receipt", "delivery",
            "hub", "storage", "transport", "confirmation", "report",
            "query", "command", "dashboard", "wizard"
        ]
        
        found_keywords = []
        for keyword in ties_keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        # Add category-specific keywords
        category_keywords = {
            "new_in_release": ["version", "release", "enhancement", "new", "feature"],
            "learning_path": ["learning", "path", "training", "foundation", "overview"],
            "how_to": ["how_to", "steps", "procedure", "guide", "process"],
            "reference": ["reference", "configuration", "settings", "specification"],
            "scheduling_app": ["scheduling", "balancing", "orders", "allocation"],
            "customer_activity": ["caw", "nominations", "shipper", "customer"],
            "fuel_calculator": ["fuel", "calculator", "evaluation", "allocation"]
        }
        
        if category in category_keywords:
            for keyword in category_keywords[category]:
                if keyword in text and keyword not in found_keywords:
                    found_keywords.append(keyword)
        
        return found_keywords

    def determine_technical_level(self, content: str, category: str) -> str:
        """Determine technical complexity level"""
        content_lower = content.lower()
        
        # Advanced indicators
        advanced_terms = ["configuration", "setup", "administrator", "technical", "advanced", "complex"]
        # Basic indicators
        basic_terms = ["overview", "introduction", "basic", "getting started", "fundamentals"]
        
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        basic_count = sum(1 for term in basic_terms if term in content_lower)
        
        if advanced_count > basic_count:
            return "advanced"
        elif basic_count > 0:
            return "basic"
        else:
            return "intermediate"

    def determine_business_impact(self, content: str, category: str) -> str:
        """Determine business impact from content"""
        content_lower = content.lower()
        
        impact_mapping = {
            "efficiency": ["streamline", "automate", "faster", "optimize"],
            "accuracy": ["accurate", "precise", "validation", "error"],
            "compliance": ["regulatory", "compliance", "audit", "standard"],
            "cost_savings": ["cost", "saving", "reduce", "efficiency"],
            "user_experience": ["user", "interface", "usability", "experience"],
            "operational": ["operation", "process", "workflow", "management"]
        }
        
        for impact, keywords in impact_mapping.items():
            if any(keyword in content_lower for keyword in keywords):
                return impact
                
        return "operational"

    def extract_learning_metadata(self, content: str, title: str) -> Dict[str, Any]:
        """Extract learning-specific metadata for learning path articles"""
        metadata = {}
        
        # Detect learning track from title
        if "merchant" in title.lower():
            metadata["learning_track"] = f"merchant_{title.lower().split(':')[-1].strip().replace(' ', '_')}"
        elif "plant" in title.lower():
            metadata["learning_track"] = f"plant_{title.lower().split(':')[-1].strip().replace(' ', '_')}"
        elif "pipeline" in title.lower() or "gathering" in title.lower():
            metadata["learning_track"] = f"pipeline_{title.lower().split(':')[-1].strip().replace(' ', '_')}"
        elif "producer" in title.lower():
            metadata["learning_track"] = f"producer_{title.lower().split(':')[-1].strip().replace(' ', '_')}"
        elif "foundation" in title.lower():
            metadata["learning_track"] = "foundations"
        
        # Determine prerequisite level
        if "foundation" in content.lower() or "basic" in content.lower():
            metadata["prerequisite_level"] = "beginner"
        elif "advanced" in content.lower() or "complex" in content.lower():
            metadata["prerequisite_level"] = "advanced"
        else:
            metadata["prerequisite_level"] = "intermediate"
            
        # Extract learning objectives
        objectives = []
        if "understand" in content.lower():
            objectives.append("understand_concepts")
        if "learn" in content.lower():
            objectives.append("learn_procedures")
        if "manage" in content.lower():
            objectives.append("manage_operations")
        if "configure" in content.lower():
            objectives.append("configure_system")
        if not objectives:
            objectives.append("general_knowledge")
            
        metadata["learning_objectives"] = objectives
        
        return metadata

    def split_content_by_strategy(self, content: str, category: str) -> List[str]:
        """Split content according to category strategy"""
        strategy = self.chunking_strategies.get(category, {"max_size": 1800, "split_pattern": r'\n\n'})
        max_size = strategy["max_size"]
        split_pattern = strategy["split_pattern"]
        
        # Split content into logical sections
        sections = re.split(split_pattern, content)
        sections = [section.strip() for section in sections if section.strip()]
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # If adding this section keeps us under limit
            if len(current_chunk) + len(section) <= max_size:
                current_chunk += "\n\n" + section if current_chunk else section
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk or split large section
                if len(section) <= max_size:
                    current_chunk = section
                else:
                    # Split large section by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= max_size:
                            temp_chunk += " " + sentence if temp_chunk else sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = sentence
                    
                    current_chunk = temp_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def create_chunk_id(self, article_id: str, chunk_index: int, content: str) -> str:
        """Create a unique chunk ID"""
        # Extract key terms for ID
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content[:100])
        key_terms = "_".join(words[:3]).lower() if words else "content"
        return f"{article_id}_{key_terms}_{chunk_index}"

    def process_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single article into chunks"""
        # Extract basic info
        article_id = str(article["id"])
        title = article["title"]
        raw_content = article["body"]
        
        # Clean content
        content = self.clean_html_content(raw_content)
        if not content or len(content) < 100:  # Skip very short content
            return []
        
        # Determine category and metadata
        category = self.determine_category(article)
        user_roles = self.determine_user_roles(content, title, category)
        keywords = self.extract_keywords(title, content, category)
        technical_level = self.determine_technical_level(content, category)
        business_impact = self.determine_business_impact(content, category)
        version = self.extract_version_info(content, title)
        
        # Split content into chunks
        content_chunks = self.split_content_by_strategy(content, category)
        
        chunks = []
        for i, chunk_content in enumerate(content_chunks):
            chunk_id = self.create_chunk_id(article_id, i, chunk_content)
            
            # Base metadata
            metadata = {
                "category": category,
                "article_id": article_id,
                "title": title,
                "user_roles": user_roles,
                "keywords": keywords,
                "technical_level": technical_level,
                "business_impact": business_impact,
                "content_length": len(chunk_content),
                "chunk_type": "ties_documentation",
                "created_at": article.get("created_at", ""),
                "updated_at": article.get("updated_at", "")
            }
            
            # Add category-specific metadata
            if category == "learning_path":
                learning_meta = self.extract_learning_metadata(content, title)
                metadata.update(learning_meta)
                metadata["sequence_order"] = i + 1
                metadata["target_role"] = user_roles[0] if user_roles else "all_users"
                
            elif category == "new_in_release":
                if "enhancement" in chunk_content.lower():
                    metadata["feature_type"] = "enhancement"
                elif "new" in chunk_content.lower():
                    metadata["feature_type"] = "new_feature"
                else:
                    metadata["feature_type"] = "update"
                    
            elif category == "how_to":
                # Extract navigation paths and steps
                nav_paths = re.findall(r'Navigate to ([^.]+)', chunk_content)
                if nav_paths:
                    metadata["navigation_path"] = nav_paths[0].strip()
                
                steps = re.findall(r'(\d+)\.\s*([^.]+\.)', chunk_content)
                if steps:
                    metadata["steps_count"] = len(steps)
                    
            elif category == "reference":
                if "configuration" in chunk_content.lower():
                    metadata["reference_type"] = "configuration"
                elif "specification" in chunk_content.lower():
                    metadata["reference_type"] = "specification"
                else:
                    metadata["reference_type"] = "general"
            
            # Determine software module
            module_mapping = {
                "trading": ["trading", "trade", "buy", "sell", "deal"],
                "scheduling": ["scheduling", "schedule", "nomination", "balance"],
                "accounting": ["accounting", "invoice", "payment", "ar", "ap"],
                "customer_activity": ["caw", "customer", "shipper", "nomination"],
                "navigation": ["launcher", "navigation", "menu", "interface"],
                "security": ["security", "authentication", "azure", "login"],
                "fuel_calculator": ["fuel", "calculator", "allocation"],
                "reporting": ["report", "query", "dashboard", "analysis"]
            }
            
            for module, terms in module_mapping.items():
                if any(term in chunk_content.lower() for term in terms):
                    metadata["software_module"] = module
                    break
            else:
                metadata["software_module"] = "general"
            
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_content,
                "metadata": metadata
            })
        
        return chunks

    def process_articles_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all articles from a folder containing individual JSON files"""
        print(f"📁 Loading articles from folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Find all JSON files in the folder
        json_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                json_files.append(os.path.join(folder_path, file))
        
        if not json_files:
            raise ValueError(f"No JSON files found in folder: {folder_path}")
        
        print(f"📊 Found {len(json_files)} JSON files to process")
        
        all_chunks = []
        successful = 0
        failed = 0
        
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            try:
                # Load individual article JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                
                # Process the article
                chunks = self.process_article(article_data)
                all_chunks.extend(chunks)
                successful += 1
                
            except Exception as e:
                file_name = os.path.basename(json_file)
                print(f"❌ Failed to process file {file_name}: {e}")
                failed += 1
        
        print(f"✅ Successfully processed {successful} articles")
        print(f"❌ Failed to process {failed} articles")
        print(f"📦 Generated {len(all_chunks)} total chunks")
        
        return all_chunks

    def process_articles_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process all articles from a JSON file"""
        print(f"📁 Loading articles from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if "articles" in data:
                articles = data["articles"]
            else:
                # Assume it's a structure with sections
                articles = []
                for section_data in data.values():
                    if isinstance(section_data, dict) and "articles" in section_data:
                        articles.extend(section_data["articles"])
        elif isinstance(data, list):
            articles = data
        else:
            raise ValueError("Unsupported JSON structure")
        
        print(f"📊 Found {len(articles)} articles to process")
        
        all_chunks = []
        successful = 0
        failed = 0
        
        for article in tqdm(articles, desc="Processing articles"):
            try:
                chunks = self.process_article(article)
                all_chunks.extend(chunks)
                successful += 1
            except Exception as e:
                print(f"❌ Failed to process article {article.get('id', 'unknown')}: {e}")
                failed += 1
        
        print(f"✅ Successfully processed {successful} articles")
        print(f"❌ Failed to process {failed} articles")
        print(f"📦 Generated {len(all_chunks)} total chunks")
        
        return all_chunks

    def upload_to_pinecone(self, chunks: List[Dict[str, Any]], 
                          index_name: str = os.getenv("PINECONE_INDEX_NAME"),
                          batch_size: int = 50):
        """Upload chunks to Pinecone using LangChain + OpenAI embeddings"""
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(index_name)
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=2048,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        print(f"🚀 Uploading {len(chunks)} chunks to Pinecone index '{index_name}'")
        
        successful_uploads = 0
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading batches"):
            batch = chunks[i:i + batch_size]
            
            try:
                # Extract texts for embedding
                texts = [chunk['content'] for chunk in batch]
                
                # Generate embeddings
                batch_embeddings = embeddings.embed_documents(texts)
                
                # Prepare vectors for Pinecone
                vectors = []
                for j, chunk in enumerate(batch):
                    vector = {
                        'id': chunk['chunk_id'],
                        'values': batch_embeddings[j],
                        'metadata': {
                            **chunk['metadata'],
                            'content': chunk['content']  # Store content in metadata
                        }
                    }
                    vectors.append(vector)
                
                # Upload to Pinecone
                index.upsert(vectors=vectors)
                successful_uploads += len(batch)
                print(f"✅ Uploaded batch {i//batch_size + 1}: {len(batch)} chunks")
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"📊 Successfully uploaded {successful_uploads}/{len(chunks)} chunks")
        
        # Verify upload
        time.sleep(3)
        stats = index.describe_index_stats()
        print(f"📈 Total vectors in index: {stats.get('total_vector_count', 0)}")
        
        return successful_uploads

def main():
    """Main function to process articles and upload to Pinecone"""
    
    # Check environment variables
    if not os.getenv('PINECONE_API_KEY'):
        print("❌ PINECONE_API_KEY environment variable not set")
        return
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize processor
    processor = TIESArticleProcessor()
    
    # Configuration
    input_path = "zendesk_docs_20250619_114622/articles_json"  # Change this to your input folder or file
    output_file = "processed_chunks.json"  # Optional: save chunks before upload
    
    try:
        # Determine if input is a folder or file
        if os.path.isdir(input_path):
            print(f"📁 Processing folder: {input_path}")
            chunks = processor.process_articles_folder(input_path)
        elif os.path.isfile(input_path):
            print(f"📄 Processing file: {input_path}")
            chunks = processor.process_articles_file(input_path)
        else:
            print(f"❌ Input path does not exist: {input_path}")
            return
        
        # Optionally save processed chunks
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved processed chunks to {output_file}")
        
        # Upload to Pinecone
        if chunks:
            processor.upload_to_pinecone(chunks)
        
        print("🎉 Process completed successfully!")
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Allow custom input path as command line argument
        input_path = sys.argv[1]
        processor = TIESArticleProcessor()
        
        # Determine if input is a folder or file
        if os.path.isdir(input_path):
            chunks = processor.process_articles_folder(input_path)
        elif os.path.isfile(input_path):
            chunks = processor.process_articles_file(input_path)
        else:
            print(f"❌ Input path does not exist: {input_path}")
            sys.exit(1)
            
        processor.upload_to_pinecone(chunks)
    else:
        main()