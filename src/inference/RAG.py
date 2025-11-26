import requests
import os
import google.generativeai as genai  # <--- CHANGED: Google AI Library
from xhtml2pdf import pisa


# --- CONFIGURATION ---
SERPER_API_KEY = "0f0678396df38b459ec2446b7680784f92d33837"
# You need a Google API Key. Get it here: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyDR_e5kOoAsdWvgSkYURpjWk6wLMZ_GGhI"

# --- HELPER FUNCTIONS ---

def search_web(query, api_key):
    print(f"🔍 Searching web for: {query}...")
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": 10}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error during web search: {e}")
        return None


def generate_html_content(search_results, api_key):
    """
    Asks Gemini 1.5 Flash to generate a styled HTML report.
    """
    print("🤖 Generating HTML report with Gemini 2.5 Flash...")

    # Configure the Google AI Client
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Prompt logic
        prompt = f"""
        You are an expert auto repair estimator.

        Here are the search results for car parts:
        {search_results}

        Task:
        1. Extract ALL price mentions.
        2. Convert them to EGP (Egyptian Pounds) if needed.
        3. Compute min, max, and average prices.
        4. Create a clean, professional HTML report.

        Requirements:
        - Use standard HTML5.
        - Include <style> tags to make it look professional (use fonts, colors, tables).
        - The table should have borders and padding.
        - Highlight the "Recommended Action" in bold/color.
        - RETURN ONLY THE HTML CODE. Start with <!DOCTYPE html> and end with </html>.
        """

        # Generate content
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"❌ Error during Gemini generation: {e}")
        return None


def convert_html_to_pdf(source_html, output_filename):
    """
    Converts HTML string to PDF file using xhtml2pdf.
    """
    print(f"⚙️ Converting HTML to PDF: {output_filename}...")

    try:
        # Open file in binary write mode
        with open(output_filename, "wb") as result_file:
            # pisa.CreatePDF parses the HTML and writes to the file
            pisa_status = pisa.CreatePDF(source_html, dest=result_file)

        if pisa_status.err:
            print("❌ Error during PDF conversion.")
        else:
            print(f"✅ Success! PDF saved as: {os.path.abspath(output_filename)}")

    except Exception as e:
        print(f"❌ File Error: {e}")


# --- MAIN EXECUTION ---

def main():
    # 1. Define query
    query_item = "Toyota Corolla headlight"
    query = f"{query_item} price Egypt"

    # 2. Search
    results = search_web(query, SERPER_API_KEY)
    if not results: return

    # 3. Generate HTML with Gemini
    # Note: Make sure to pass the GEMINI_API_KEY here
    html_content = generate_html_content(results, GEMINI_API_KEY)
    if not html_content: return

    # Cleanup: Gemini usually wraps code in markdown blocks. We remove them.
    if "```html" in html_content:
        html_content = html_content.split("```html")[1].split("```")[0]
    elif "```" in html_content:
        html_content = html_content.split("```")[1].split("```")[0]

    # 4. Convert to PDF
    pdf_filename = "damage_report.pdf"
    convert_html_to_pdf(html_content, pdf_filename)

