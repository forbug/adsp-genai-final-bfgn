import pandas as pd
import re
import html

def prepare_amazon_for_embeddings(data, from_csv=False):
    """
    Cleans Amazon product metadata for vector embeddings (CLIP + ChromaDB).
    Removes HTML, normalizes categories/prices, deduplicates repeated sentences.
    
    Args:
        data (pd.DataFrame or str): DataFrame or path to CSV file.
        from_csv (bool): Set True if passing a CSV file path.
        
    Returns:
        pd.DataFrame: Cleaned data with a 'search_text' column.
    """
    
    # Load if CSV path provided
    if from_csv:
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    # Relevant columns
    useful_cols = [
        "Uniq Id",
        "Product Name",
        "Category",
        "About Product",
        "Technical Details",
        "Image"
    ]
    embed_df = df[[col for col in useful_cols if col in df.columns]]
    
    # --- Cleaning Helpers ---
    def clean_text(text):
        """Basic text normalization and HTML cleanup."""
        if pd.isna(text):
            return ""
        text = html.unescape(str(text))
        text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
        text = text.replace("|", ", ")        # category separator
        text = re.sub(r"\s+", " ", text)      # normalize spaces
        return text.strip().lower()
    
    def normalize_price(price):
        """Extract numeric price and format uniformly."""
        if pd.isna(price):
            return ""
        match = re.search(r"[\d,.]+", str(price))
        if match:
            try:
                val = float(match.group(0).replace(",", ""))
                return f"price: {val:.2f} USD"
            except:
                return ""
        return ""
    
    def clean_category(cat):
        """Convert category hierarchy into natural text."""
        cat = clean_text(cat)
        cat = re.sub(r",\s*,", ",", cat)  # fix double commas
        return cat
    
    def deduplicate_sentences(*texts):
        """Merge multiple text fields into deduplicated sentence list."""
        combined = " ".join([t for t in texts if t])
        # Split on sentence-like boundaries
        parts = re.split(r"(?<=[.!?])\s+|,|\n", combined)
        seen = set()
        deduped = []
        for part in parts:
            p = part.strip()
            if p and p not in seen:
                seen.add(p)
                deduped.append(p)
        return " ".join(deduped)
    
    # --- Apply cleaning ---
    for col in ["Product Name", "About Product", "Product Specification", "Technical Details", "Model Number"]:
        if col in embed_df.columns:
            embed_df[col] = embed_df[col].apply(clean_text)
    
    if "Category" in embed_df.columns:
        embed_df["Category"] = embed_df["Category"].apply(clean_category)
    
    # --- Build final search_text ---
    def build_search_text(row):
        parts = []
        if "Product Name" in row and row["Product Name"]:
            parts.append(f"name: {row['Product Name']}")
        if "Category" in row and row["Category"]:
            parts.append(f"category: {row['Category']}")
        if "Selling Price" in row and row["Selling Price"]:
            parts.append(row["Selling Price"])
        
        # Merge About, Specs, Details into deduplicated form
        if any(row.get(col) for col in ["About Product", "Product Specification", "Technical Details"]):
            merged = deduplicate_sentences(
                row.get("About Product", ""),
                row.get("Technical Details", "")
            )
            if merged:
                parts.append(f"description: {merged}")
        
        return " | ".join(parts)
    
    df["search_text"] = embed_df.apply(build_search_text, axis=1)
    df["image_url"] = df['Image'].str.split("|").str[0]
    df['id'] = range(len(df))
    df["product_id"] = df['Uniq Id']
    df["name"] = df['Product Name']
    
    return df

if __name__ == "__main__":
    cleaned_df = prepare_amazon_for_embeddings("data/amazon_product_data_2020.csv", from_csv=True)
    cleaned_df.to_csv("data/amazon_products_cleaned.csv", index=False)
