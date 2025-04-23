import http.server
import socketserver
import json
import urllib.parse
import os
import sys
import re # Import regex module for formatting

# Define the port to run the server on
PORT = 8001
# Define the path to the debate JSON file (will be set via command line arg)
DEBATE_FILE_PATH = None

class DebateHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler to serve the debate viewer.
    """
    def do_GET(self):
        """
        Handle GET requests. Parses the URL for the question key and serves
        the corresponding debate HTML page.
        """
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Get the question key from the query parameters
        question_keys_in_url = query_params.get('q', [])
        if not question_keys_in_url:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Error: Please provide a 'q' parameter in the URL with the question key.")
            return

        question_key = question_keys_in_url[0]

        # Load the debate data
        if not os.path.exists(DEBATE_FILE_PATH):
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error: Debate file not found at {DEBATE_FILE_PATH}".encode('utf-8'))
            return

        try:
            with open(DEBATE_FILE_PATH, 'r', encoding='utf-8') as f:
                debate_data = json.load(f)
        except json.JSONDecodeError:
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Error: Could not parse the debate JSON file.")
            return
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error reading debate file: {e}".encode('utf-8'))
            return

        # Check if the question key exists
        questions = debate_data.get('prompt', {}).get('questions', {})
        if question_key not in questions:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error: Question key '{question_key}' not found in the debate data.".encode('utf-8'))
            return

        initial_question = questions[question_key]
        debate_for_question = debate_data.get('debate', {}).get(question_key, {})
        llm_meta = debate_data.get('meta', {}).get('llms', {})
        llm_keys = list(llm_meta.keys())

        # Determine the maximum number of rounds for this question
        num_rounds = 0
        if debate_for_question:
             # Find the maximum number of responses for any LLM in this question
            num_rounds = max([len(llm_data.get('responses', [])) for llm_data in debate_for_question.values()], default=0)


        # Generate the HTML content
        html_content = self.generate_html(initial_question, debate_for_question, llm_meta, llm_keys, num_rounds)

        # Send the response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def format_text(self, text):
        """
        Applies minimal formatting: newline to <br />, **text** to <b>text</b>, *text* to <i>text</i>.
        """
        # Replace newlines with <br />
        formatted_text = text.replace('\n', '<br />')

        # Replace **text** with <b>text</b>
        # Use regex to handle cases where ** might contain newlines or other characters
        formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_text)

        # Replace *text* with <i>text</i>
        # Use regex, and be careful not to match the content within <b> tags
        # This regex attempts to match *...* only if it's not inside <b> tags.
        # It's a simplified approach and might not cover all edge cases.
        formatted_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_text)

        # Replace <think> with <span class="think">text</span>
        formatted_text = re.sub(r'<think>(.*?)</think>', r'<span class="think">\1</span>', formatted_text)

        return formatted_text


    def generate_html(self, initial_question, debate_for_question, llm_meta, llm_keys, num_rounds):
        """
        Generates the HTML content for the debate viewer.
        """
        # Define grid columns: one for round number + one for each LLM
        grid_template_columns = f"auto repeat({len(llm_keys)}, 1fr)"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Debate Viewer - Question: {initial_question[:50]}...</title>
    <style>
        body {{
            font-family: sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
            color: #333;
        }}
        .container {{
            display: grid;
            grid-template-columns: {grid_template_columns};
            gap: 15px;
            width: 100%;
            max-width: 1200px; /* Limit max width */
            margin: 0 auto; /* Center the grid */
            padding: 10px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }}
        .grid-item {{
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
            word-wrap: break-word; /* Prevent long words from overflowing */
            overflow-wrap: break-word; /* Prevent long words from overflowing */
            display: flex; /* Use flexbox for content alignment */
            flex-direction: column; /* Stack full and summary vertically */
        }}
         .grid-item hr {{
            border-top: 1px dashed #ccc;
            margin: 10px 0; /* Add some spacing around the dashed line */
        }}
        .header {{
            font-weight: bold;
            text-align: center;
            background-color: #e9e9e9;
            padding: 10px;
            border-radius: 5px;
        }}
        .initial-question {{
            grid-column: 1 / span {len(llm_keys) + 1}; /* Span across all columns */
            font-style: italic;
            margin-bottom: 20px;
            padding: 15px;
            border: 1px dashed #ccc;
            background-color: #fffacd; /* Light yellow background */
            border-radius: 8px;
        }}
        .round-label {{
            font-weight: bold;
            text-align: center;
            padding: 15px;
            background-color: #e0e0e0;
            border-radius: 5px;
        }}
        .full-response {{
            flex-grow: 1; /* Allow full response to take up available space */
        }}
        .summary-response {{
            color: #F00; /* Slightly muted color for summary */
        }}
        
        .think {{
            font-size: 0.75em; /* Smaller font for summary */
            color: #999; /* Grey color for think */
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="initial-question">
            <h2>Question:</h2>
            <p>{self.format_text(initial_question)}</p>
        </div>

        <div class="header">Round</div> """

        # Add headers for each LLM column
        for llm_key in llm_keys:
            display_name = llm_meta.get(llm_key, {}).get('llm_display_name', llm_key)
            html += f'        <div class="header">{display_name}</div>\n'

        # Add rows for each round
        for round_idx in range(num_rounds):
            # Round label cell
            html += f'        <div class="round-label">Round {round_idx + 1}</div>\n'

            # LLM response cells for this round
            for llm_key in llm_keys:
                full_response_text = "No response in this round."
                summary_response_text = ""
                llm_debate_data = debate_for_question.get(llm_key, {})
                if len(llm_debate_data.get('responses', [])) > round_idx:
                    full_response_text, summary_response_text = llm_debate_data['responses'][round_idx]

                # Apply formatting
                formatted_full_response = self.format_text(full_response_text)
                formatted_summary_response = self.format_text(summary_response_text)

                html += f'        <div class="grid-item llm-response">\n'
                html += f'            <div class="full-response">{formatted_full_response}</div>\n'
                if summary_response_text: # Only add summary and HR if summary exists
                     html += f'            <hr>\n'
                     html += f'            <div class="summary-response">{formatted_summary_response}</div>\n'
                html += f'        </div>\n'

        html += """
    </div>
</body>
</html>
"""
        return html

# Main part of the script to run the server
if __name__ == "__main__":
    # Parse command line arguments for the debate file path
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_debate_file.json>")
        sys.exit(1)

    DEBATE_FILE_PATH = sys.argv[1]

    if not os.path.exists(DEBATE_FILE_PATH):
        print(f"Error: Debate file not found at {DEBATE_FILE_PATH}")
        sys.exit(1)

    # Load data once to get a valid question key for the example URL
    try:
        with open(DEBATE_FILE_PATH, 'r', encoding='utf-8') as f:
            initial_data_load = json.load(f)
            question_keys = initial_data_load.get('prompt', {}).get('questions', {}).keys()
            example_question_key = next(iter(question_keys), None) # Get the first key or None
    except Exception:
        example_question_key = "example_question_key" # Fallback if file can't be read or is empty

    # Set up the server
    handler = DebateHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving debate viewer from {DEBATE_FILE_PATH} at http://localhost:{PORT}/")
        print("Append '?q=<question_key>' to the URL to view a specific question.")
        if example_question_key:
            print(f"Example: http://localhost:{PORT}/?q={example_question_key}")
        else:
             print(f"Could not find any question keys in the file to provide an example URL.")


        # Serve forever
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

