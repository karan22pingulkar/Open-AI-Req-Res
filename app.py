import os
import asyncio
from datetime import datetime
from flask import Flask, request, jsonify
import httpx
from openai import OpenAI, AsyncOpenAI
from db import db

app = Flask(__name__)


proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

if proxy_url:

    sync_http_client = httpx.Client(proxies=proxy_url)
    openai_client = OpenAI(http_client=sync_http_client)

    async_http_client = httpx.AsyncClient(proxies=proxy_url)
    async_openai_client = AsyncOpenAI(http_client=async_http_client)
else:
    openai_client = OpenAI()
    async_openai_client = AsyncOpenAI()


@app.route("/", methods=["GET"])
def home():
    try:
        collections = db.list_collection_names()
        return jsonify({
            "status": "healthy",
            "database_connected": True,
            "existing_collections": collections
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/prompt/single", methods=["POST"])
def single_prompt():
    data = request.get_json()
    if not data or "userInput" not in data:
        return jsonify({"error": "Missing 'userInput' in request body"}), 400

    user_input = data["userInput"]
    prompt_doc = db.prompts.find_one({"_id": "Education_Prompt"})
    if not prompt_doc:
        return jsonify({"error": "Prompt template not found"}), 404

    final_prompt = prompt_doc["template"].replace("{{userInput}}", user_input)

    try:
        try:
            api_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_prompt}]
            )
            ai_text = api_response.choices[0].message.content
        except Exception as api_err:
            if "insufficient_quota" in str(api_err):
                ai_text = f"[MOCK AI RESPONSE] Answer for: {user_input}"
            else:
                raise api_err

        # Log History

        db.history.insert_one({
            "prompt_id": "Education_Prompt",
            "user_input": user_input,
            "final_prompt": final_prompt,
            "ai_response": ai_text,
            "timestamp": datetime.utcnow()
        })

        return jsonify({"response": ai_text}), 200
    except Exception as err:
        return jsonify({"error": str(err)}), 500


async def process_single_bulk_item(user_input, template_string):
    """Asynchronous worker that processes a single prompt string independently."""
    final_prompt = template_string.replace("{{userInput}}", user_input)
    try:

        response = await async_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}]
        )
        ai_text = response.choices[0].message.content
    except Exception as err:

        ai_text = f"[MOCK ASYNC RESPONSE] Comprehensive analysis for: '{user_input}'"

    # Log individual item history tracking into MongoDB

    history_log = {
        "prompt_id": "Education_Prompt_Bulk",
        "user_input": user_input,
        "final_prompt": final_prompt,
        "ai_response": ai_text,
        "timestamp": datetime.utcnow()
    }

    check_print = db.history.insert_one(history_log)
    print(f"Mongo successful:{check_print.inserted_id}")
    return ai_text


@app.route("/api/prompt/bulk", methods=["POST"])
def bulk_prompt():
    data = request.get_json()

    if not data or "userInputs" not in data or not isinstance(data["userInputs"], list):
        return jsonify({"error": "Missing list 'userInputs' in request body"}), 400

    user_inputs = data["userInputs"]

    prompt_doc = db.prompts.find_one({"_id": "Education_Prompt"})
    if not prompt_doc:
        return jsonify({"error": "Base prompt template not found"}), 404

    template_string = prompt_doc["template"]

    # inner
    async def run_parallel_tasks():

        tasks = [process_single_bulk_item(
            inp, template_string) for inp in user_inputs]

        return await asyncio.gather(*tasks)

    ordered_responses = asyncio.run(run_parallel_tasks())

    return jsonify({
        "responses": ordered_responses
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
