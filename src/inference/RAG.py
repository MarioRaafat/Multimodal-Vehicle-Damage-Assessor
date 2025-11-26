import requests
import os
import json
import google.generativeai as genai
from groq import Groq
from xhtml2pdf import pisa
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- CONFIGURATION ---
# Now we fetch them securely from the environment
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if keys are missing (Optional safety step)
if not all([SERPER_API_KEY, GEMINI_API_KEY, GROQ_API_KEY]):
    raise ValueError("Error: Missing API keys. Please check your .env file.")


# --- 1. REASONING LAYER (DeepSeek R1 via Groq) ---

def decide_repair_strategy(car_info, part_name, damage_type, severity, api_key):
    """
    Analyzes damage for a SINGLE part.
    """
    print(f"🧠 DeepSeek R1: Analyzing logic for {part_name}...")

    client = Groq(api_key=api_key)

    prompt = f"""
    You are a Senior Automotive Adjuster. 
    Analyze the following case to decide between REPAIR or REPLACE.

    Vehicle: {car_info}
    Part: {part_name}
    Damage Type: {damage_type}
    Severity: {severity}

    Logic Rules:
    1. Safety parts (glass, airbags, seatbelts) are ALWAYS replace.
    2. Severe structural damage is replace.
    3. Minor cosmetic damage (dents, scratches) on metal/plastic is repair.
    4. Complex electronics (headlights with cracked lens) are usually replace.

    Task:
    1. Make a decision: "Repair" or "Replace".
    2. Write a short reasoning (1 sentence).
    3. Generate a highly optimized Google Search Query for Egypt.

    OUTPUT FORMAT:
    Return ONLY a raw JSON object.
    {{
        "decision": "Replace",
        "reasoning": "Explanation here.",
        "search_query": "Toyota Corolla 2022 {part_name} price Egypt"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Using Llama as proxy for R1 reasoning
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        content = response.choices[0].message.content
        if "</think>" in content: content = content.split("</think>")[-1].strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    except Exception as e:
        print(f"❌ Error in Reasoning Layer for {part_name}: {e}")
        return {
            "decision": "Manual Check",
            "reasoning": "AI Failed",
            "search_query": f"{car_info} {part_name} price Egypt"
        }


# --- 2. SEARCH LAYER ---

def search_web(query, api_key):
    print(f"🔍 Searching web for: {query}...")
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": 5}  # Limit to 5 results per part to keep it fast
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error during web search: {e}")
        return {}


# --- 3. REPORT LAYER (Gemini) ---

def generate_final_report(car_info, all_parts_data, api_key):
    """
    Takes a LIST of parts data and generates ONE consolidated report.
    """
    print("🤖 Gemini 2.5: Writing consolidated report...")

    try:
        genai.configure(api_key=api_key)
        # Using 1.5 Flash (Standard stable version)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # We convert the list of dicts to a string for the LLM to read
        data_str = json.dumps(all_parts_data, indent=2)

        prompt = f"""
        You are an expert auto repair estimator.

        VEHICLE: {car_info}

        I have analyzed multiple parts for this vehicle. 
        Here is the JSON data containing the Decision, Logic, and Market Search Results for EACH part:

        {data_str}

        Task:
        Generate a professional HTML repair estimate report that covers ALL parts.

        Structure Requirements:
        1. **Header**: Vehicle Info and Date.
        2. **Executive Summary Table**: A table listing Part Name, Damage, Decision (Repair/Replace), and Estimated Cost Range (calculated from search results).
        3. **Detailed Breakdown**: Create a separate section for each part that includes:
           - The Analysis Logic (Why did we decide to Repair/Replace?).
           - Market Data Table (List specific items/prices found in the search results).
        4. **Grand Total**: An estimated total range for the whole job.

        Styling Requirements:
        - Use standard HTML5 with internal CSS.
        - Font: Helvetica or Arial.
        - Color Scheme: Navy Blue headers, light gray backgrounds for sections.
        - Use color badges for decisions (Red = Replace, Green = Repair).
        - RETURN ONLY RAW HTML.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"❌ Error during Gemini generation: {e}")
        return None


# --- 4. PDF CONVERSION ---

def convert_html_to_pdf(source_html, output_filename):
    print(f"⚙️ Converting Report to PDF...")
    try:
        with open(output_filename, "wb") as result_file:
            pisa_status = pisa.CreatePDF(source_html, dest=result_file)

        if not pisa_status.err:
            print(f"✅ Success! PDF saved as: {os.path.abspath(output_filename)}")
        else:
            print("❌ Error during PDF conversion.")
    except Exception as e:
        print(f"❌ File Error: {e}")


# --- 5. ORCHESTRATOR (PROCESS MULTIPLE PARTS) ---

def process_full_case(car_info, parts_list):
    """
    Iterates through all parts, gathers data, and generates one report.
    """

    # This list will store the full bundle of data for every part
    full_case_data = []

    print(f"🚗 Starting Full Case Analysis for: {car_info}")
    print(f"📦 Parts to analyze: {len(parts_list)}")

    # --- LOOP THROUGH EACH PART ---
    for part in parts_list:
        part_name = part['name']
        damage_type = part['damage']
        severity = part['severity']

        print(f"\n--- Processing Part: {part_name} ---")

        # 1. Decide Strategy
        strategy = decide_repair_strategy(
            car_info, part_name, damage_type, severity, GROQ_API_KEY
        )
        print(f"   👉 Decision: {strategy.get('decision')}")

        # 2. Search Web
        search_results = search_web(strategy.get('search_query', ''), SERPER_API_KEY)

        # 3. Bundle Data
        part_record = {
            "part_name": part_name,
            "damage_input": f"{damage_type} ({severity})",
            "strategy": strategy,
            "market_data": search_results
        }

        full_case_data.append(part_record)

    # --- GENERATE REPORT ---
    print("\n📝 Generating Final PDF...")
    html_content = generate_final_report(car_info, full_case_data, GEMINI_API_KEY)

    if html_content:
        # Cleanup Markdown
        if "```html" in html_content:
            html_content = html_content.split("```html")[1].split("```")[0]
        elif "```" in html_content:
            html_content = html_content.split("```")[1].split("```")[0]

        convert_html_to_pdf(html_content, "full_damage_report.pdf")


# --- MAIN EXECUTION ---

def main():
    # Example Input Data (This would come from your UI or YOLO loop)
    car_info = "Toyota Corolla 2022"

    # List of detected parts and their status
    parts_to_process = [
        {
            "name": "Front Bumper",
            "damage": "Deep Scratch",
            "severity": "Moderate"
        },
        {
            "name": "Headlight",
            "damage": "Cracked Lens",
            "severity": "Severe"
        },
        {
            "name": "Hood",
            "damage": "Minor Dent",
            "severity": "Low"
        }
    ]

    # Run the main process
    process_full_case(car_info, parts_to_process)


if __name__ == "__main__":
    main()