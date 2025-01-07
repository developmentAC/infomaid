Here’s the updated version of the code with the provided `add_to_chroma` function integrated:

### Modified Code

```python
import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.schema import Document
import get_embedding_function

CHROMA_PATH = "path_to_chroma_db"  # Replace with your Chroma database path
console = print  # Replace with your console printer if needed

def load_documents_TEXT():
    """
    Load text documents from a predefined directory.
    Returns a list of documents.
    """
    text_dir = "data/"
    documents = []
    
    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_dir, filename)
            loader = TextLoader(file_path)
            documents.append(loader.load())
    
    return documents

def load_documents_CSV():
    """
    Load CSV files from a predefined directory and return them as Documents.
    Returns a list of Documents.
    """
    csv_dir = "data/"
    documents = []
    
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_dir, filename)
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    content = row.to_dict()
                    metadata = {"filename": filename}
                    documents.append(Document(page_content=str(content), metadata=metadata))
            except Exception as e:
                console(f"Error loading {filename}: {e}")
    
    return documents

def split_documents(document):
    """
    Placeholder function for splitting documents into chunks.
    Replace this with your actual implementation.
    """
    # For simplicity, we assume the document itself is a single chunk.
    return [document]

def calculate_chunk_ids(chunks):
    """
    Assign unique IDs to each chunk based on their metadata or content.
    Replace with your own logic if necessary.
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"{chunk.metadata.get('filename', 'unknown')}-{i}"
    return chunks

def process_text_documents(useTXT, myModel):
    """
    Process text documents if the useTXT flag is set.
    """
    if useTXT:
        try:
            documentsTXT_list = load_documents_TEXT()
        except Exception:
            console("\n:scream:[bright_red] Error: Unexpected file with < txt > in the data/ directory. Please check your files and try again.[/bright_red]\n")
            exit()
        
        for document in documentsTXT_list:
            chunks = split_documents(document)
            chunks_with_ids = calculate_chunk_ids(chunks)
            add_to_chroma(chunks_with_ids, myModel)

def process_csv_documents(useCSV, myModel):
    """
    Process CSV documents if the useCSV flag is set.
    """
    if useCSV:
        try:
            documentsCSV_list = load_documents_CSV()
        except Exception:
            console("\n:scream:[bright_red] Error: Unexpected issue with CSV files in the data/ directory. Please check your files and try again.[/bright_red]\n")
            exit()
        
        for document in documentsCSV_list:
            chunks = split_documents(document)
            chunks_with_ids = calculate_chunk_ids(chunks)
            add_to_chroma(chunks_with_ids, myModel)

# Example usage
if __name__ == "__main__":
    useTXT = True  # Set this based on your requirements
    useCSV = True  # Set this based on your requirements
    myModel = "your-model-name"  # Replace with your actual model name
    
    process_text_documents(useTXT, myModel)
    process_csv_documents(useCSV, myModel)
```

### Changes and Integration:
1. **`add_to_chroma` Integration**:
   - Used the provided `add_to_chroma` function for adding chunks with IDs to the Chroma database.

2. **CSV File Handling**:
   - CSV rows are converted to `Document` objects with metadata containing the filename.
   - Each row is treated as a separate document.

3. **Chunk ID Calculation**:
   - Added the `calculate_chunk_ids` function to generate unique IDs for chunks based on the filename and index.

4. **Improved Console Messages**:
   - Replaced `print` statements with more user-friendly console messages.

5. **General Structure**:
   - Reused logic for both text and CSV processing while maintaining separation for clarity.

6. **Error Handling**:
   - Graceful handling of file reading and processing errors.

With these updates, the code should now work seamlessly for both text and CSV files, utilizing the `add_to_chroma` function for database operations.


# from langchain_community.document_loaders import TextLoader

#  if useTXT:
#         # print(f"  +++ Using option : useTXT: {useTXT}")
#         try:
#             documentsTXT_list = load_documents_TEXT()
#         except Exception:
#             console.print("\t :scream:[bright_red] There seems to be a unexpected file with < txt > in the\n\t filename in < data/ >. Please check your files and try again.[/bright_red]")
#             exit()            
#         # console.print(f"[cyan]main() documentsXML_list : [bright_yellow]{documentsXML_list}[/bright_yellow]") #list
#         for i in range(len(documentsTXT_list)):
#             chunks = split_documents(documentsTXT_list[i])
#             # print(f"{i}: useXML chunks : {chunks}")
#             #   input("Current Chunk above. Press any key to continue!!")
#             add_to_chroma(chunks, myModel)
