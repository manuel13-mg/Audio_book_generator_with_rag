import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import sys
import traceback
from dotenv import load_dotenv

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "my_vectordb")

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def query_rag_chat(user_question):
    """
    Restructured RAG Function with strict formatting rules:
    - No bold text (**).
    - clear paragraph structure.
    - Professional tone.
    """
    if not API_KEY:
        return "Configuration Error: GEMINI_API_KEY not found in .env file."

    try:
        # --- STEP 1: CONNECT TO DB ---
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            collection = client.get_collection(name="document_chunks")
        except Exception:
            return "Database Error: No book has been indexed yet. Please upload a file first."

        # --- STEP 2: EMBED QUESTION ---
        try:
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = embed_model.encode([user_question]).tolist()
        except Exception as e:
            return f"Embedding Error: {e}"

        # --- STEP 3: SEARCH DB ---
        results = collection.query(query_embeddings=query_embedding, n_results=5)
        
        if not results['documents'] or not results['documents'][0]:
            return "I searched the document but couldn't find any relevant information."

        retrieved_chunks = results['documents'][0]
        context_block = "\n\n".join(retrieved_chunks)

        # --- STEP 4: STRICT SYSTEM PROMPT (The Fix) ---
        rag_prompt = f"""
        You are a professional technical assistant. Your goal is to answer the user's question clearly and concisely using the provided Context.

        STRICT FORMATTING RULES:
        1. Do NOT use bold text (do not use asterisks like **text**).
        2. Do NOT use bullet points or lists. Write in full, flowing paragraphs.
        3. Structure your response into clear sections using line breaks if necessary.
        4. Start with a direct answer to the question, then provide details.
        5. Maintain a professional, clean tone.
        
        Context:
        {context_block}
        
        User Question:
        {user_question}
        """

        # --- STEP 5: MODEL SELECTION ---
        try:
            valid_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    valid_models.append(m.name)
            
            # Filter: Prefer 'flash', Avoid 'exp'
            stable_models = [m for m in valid_models if 'exp' not in m]
            if not stable_models: stable_models = valid_models 
            
            # Select best model
            selected_model = next((m for m in stable_models if 'flash' in m), None)
            if not selected_model:
                selected_model = next((m for m in stable_models if 'pro' in m), stable_models[0])

            # Generate
            model = genai.GenerativeModel(selected_model)
            response = model.generate_content(rag_prompt)
            
            # Extra safety cleanup to remove any stray markdown
            clean_text = response.text.replace("**", "").replace("* ", "")
            return clean_text.strip()

        except Exception as e:
            print(f"❌ Model Generation Error: {e}")
            return f"AI Error: {str(e)}"

    except Exception as e:
        print(f"❌ RAG ERROR: {e}")
        traceback.print_exc()
        return f"System Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(query_rag_chat(" ".join(sys.argv[1:])))
    else:
        print("Usage: python rag.py <question>")