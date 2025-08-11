
# ADSP 32027 Generative AI Principles SU2025 - Final Assignment

## Quickstart: Running the UI

Follow these steps to set up your environment, preprocess the data, create all necessary indexes, and launch the Streamlit UI:

1. **Clone the repository and set up the Python environment:**
   ```bash
   git clone git@github.com:forbug/adsp-genai-final-bfgn.git
   cd adsp-genai-final-bfgn
   poetry config virtualenvs.in-project true
   poetry env use 3.11.12
   source .venv/bin/activate
   poetry install
   ```

2. **Environment Variables**

   Copy `.env.example` to `.env` and fill in any required secrets or configuration values.

3. **Preprocess the product data:**
   ```bash
      python scripts/preprocess_product_data.py
   ```

4. **Create all Chroma indexes:**
   - **Multimodal index:**
     ```bash
     python scripts/create_multimodal_index.py
     ```
   - **Text-only index:**
     ```bash
     python scripts/create_text_only_index.py
     ```
   - **Image-only index:**
     ```bash
     python scripts/create_image_only_index.py
     ```

5. **Run the Streamlit UI:**
   ```bash
   streamlit run ui/app.py
   ```

You can now interact with the Amazon Product Chatbot in your browser.

---

## Troubleshooting

#### Configure Git (Optional)

1. Add your SSH keys to your GitHub account.
   * Follow the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
2. Configure your git username and email:
   ```bash
   git config user.name "<your-username>"
   git config user.email "<your-github-email>"
   ```
3. Clone the repository:
   ```bash
   git clone git@github.com:forbug/adsp-genai-final-bfgn.git
   ```

