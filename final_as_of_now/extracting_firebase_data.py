import json
import chromadb
from chromadb.utils import embedding_functions

# CODE TO EXTRACT DATA THAT WAS STORED IN FIREBSE WHICH WAS DOWNLOADED AND PASSED AS A JSON FILE.
# THE DATA IS THEN FLATTENED AND PUSHED TO CHROMA DB

# Initialize Chroma Client
client = chromadb.PersistentClient(path='./MasterDatabase')

# Load JSON data
with open("./data.json", "r") as file:
    data = json.load(file)

# Function to flatten nested dictionaries
def flatten_dict(d, parent_key='', sep='__'):
    """
    Flattens a nested dictionary into a single-level dictionary.
    Nested keys are joined with `sep`.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Define the collection
collection = client.get_or_create_collection(name="stock_data")

# Process the JSON data
for company, details in data.items():
    # Flatten the nested metadata
    metadata = flatten_dict({
        "industry_classification": details.get("Detailed Industry Classification", {}),
        "high_lows": details.get("Detailed High Lows", {}),
        "previous_day": details.get("Previous Day", {}),
        "price_bands": details.get("Price Bands", {}),
        "market_capitalization": details.get("Market Capitalization", {}),
        "financials": details.get("Financials", {}),
        "classification": details.get("Classification", {}),
        "peer_comparison": details.get("Peer Comparison", {}),
        "quarterly_results": details.get("Quarterly_Results", {}),
        "report_url": details.get("report_url", "")
    })

    # Add data to the collection
    collection.add(
        ids=[company],  # Use the company name as ID
        embeddings=[[0.0] * 128],  # Placeholder embedding
        metadatas=[metadata],  # Flattened metadata
        documents=[f"Data for {company}"]  # Optional document field
    )

print("Data successfully pushed to ChromaDB!")

# Get company data
# Function to unflatten dictionary (reverse of flatten_dict)
def unflatten_dict(d, sep='__'):
    """
    Unflattens a single-level dictionary back into a nested dictionary.
    Keys split by `sep` become nested structures.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result

# Function to get company data
def get_company_data(company_name):
    """
    Retrieves data for a specific company from ChromaDB.
    
    Args:
        company_name (str): Name of the company to query
        
    Returns:
        dict: Company data in nested format
    """
    # Query the collection
    result = collection.get(
        ids=[company_name],
        include=['metadatas', 'documents']
    )
    
    if not result['ids']:
        return f"No data found for company: {company_name}"
    
    # Unflatten the metadata
    unflattened_data = unflatten_dict(result['metadatas'][0])
    return unflattened_data

# Example usage
company_name = "INFY"  # Replace with your company name
company_data = get_company_data(company_name)

# Display results using pretty print
from pprint import pprint
print(f"\nData for {company_name}:")
pprint(company_data)