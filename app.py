import os
from datetime import datetime
from flask import Flask, request, jsonify
import httpx
from openai import OpenAI
from db import db

app = Flask(__name__)

# --- Secure OpenAI Client Initialization ---
# Checks for proxy environment variables to prevent initialization crashes.
proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

if proxy_url:
    http_client = httpx.Client(proxies=proxy_url)
    openai_client = OpenAI(http_client=http_client)
else:
    openai_client = OpenAI()


# --- Verification Route ---
@app.route("/", methods=["GET"])
def home():
    try:
        collections = db.list_collection_names()
        return jsonify({
            "status": "healthy",
            "message": "Flask server is running perfectly!",
            "database_connected": True,
            "existing_collections": collections
        }), 200
    except Exception as e:
        return jsonify({
            "status": "partial_success",
            "message": "Flask server is running, but database connection failed.",
            "error": str(e)
        }), 500


# --- Steps 1 to 5: Single Prompt Processing Endpoint ---
@app.route("/api/prompt/single", methods=["POST"])
def single_prompt():
    # 1. Capture and validate incoming JSON payload
    data = request.get_json()
    if not data or "userInput" not in data:
        return jsonify({"error": "Missing 'userInput' in request body"}), 400

    user_input = data["userInput"]

    # 2. Fetch the prompt template from MongoDB Atlas
    prompt_doc = db.prompts.find_one({"_id": "Education_Prompt"})
    if not prompt_doc:
        return jsonify({
            "error": "Prompt template 'Education_Prompt' not found in database. Please seed your database."
        }), 404

    # 3. Perform string interpolation (replace the placeholder)
    raw_template = prompt_doc["template"]
    final_prompt = raw_template.replace("{{userInput}}", user_input)

    # 4. Handle API Call with Mock Fallback for Development Stability
    try:
        try:
            # Attempt to hit the live OpenAI API
            api_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": final_prompt}
                ]
            )
            ai_text = api_response.choices[0].message.content

        except Exception as api_err:
            # Check if the error message contains the specific quota issue
            if "insufficient_quota" in str(api_err):
                print(
                    "⚠️ OpenAI Quota exceeded! Falling back to Mock responses for testing.")
                # We inject standard, structured text to simulate OpenAI's response
                ai_text = (
                    f"[MOCK AI RESPONSE] To pass the CA Final exam, you must secure "
                    f"a minimum of 40% marks in each individual paper and an aggregate "
                    f"of 50% total marks across all papers in a group."
                )
            else:
                # If it's a completely different error (network failure, bad key), raise it up
                raise api_err

        # 5. Build and log the complete execution document inside the 'history' collection
        history_log = {
            "prompt_id": "Education_Prompt",
            "user_input": user_input,
            "final_prompt": final_prompt,
            "ai_response": ai_text,
            "timestamp": datetime.utcnow()
        }
        db.history.insert_one(history_log)
        # insert_result = db.history.insert_one(history_log)
        # print(
        #     f"🚀 MongoDB Write Successful! Document ID: {insert_result.inserted_id}")
        # print(f"🚀 Collection Name in Use: history")

        # 6. Return response to client in exact requested format
        return jsonify({
            "response": ai_text
        }), 200

    except Exception as err:
        return jsonify({
            "error": "An error occurred during execution or database logging.",
            "details": str(err)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
