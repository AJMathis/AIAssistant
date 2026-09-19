from openai import OpenAI, APIStatusError, APIConnectionError
import os
import json
import time
import datetime
import requests

class CanvasAgent:
    api_key = ""
    client = ""
    messages = ""
    tools = ""
    AVAILABLE_FUNCTIONS = ""

    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.messages = [{"role": "system", "content": self.buildSystemPrompt()},{"role": "user", "content": "Hello, I am Jarvis"}]
        self.tools = self.get_tools()
        self.AVAILABLE_FUNCTIONS = {
                "get_canvas_courses": lambda args: self.get_canvas_courses(os.environ.get("CANVAS_TOKEN")),
                "get_canvas_assignments": lambda args: self.get_canvas_assignments(os.environ.get("CANVAS_TOKEN"), args.get("courseId")),
            }


    #HELPER METHODS#

    def buildSystemPrompt(self):
        with open("canvasAgentPrompt.txt", "r", encoding="utf-8") as f:
            return f"Today is {datetime.datetime.now()}. {f.read().strip()}"

    def try_again_loop(self, func):
        max_attempts = 3
        time_increase = 2
        for attempt in range(max_attempts+1):
            wait_time = 2 * attempt
            if(attempt == max_attempts):
                raise RuntimeError("Failed to reach server. Out of Attempts.")
            try:
                result = func()
                return result
            except (APIStatusError) as e: ##ADD MORE EXCEPTION HANDLING
                if(e.status_code == 503):
                    print(f"Failed to reach server. Trying again. (Attempt {attempt + 1}/3)")
                    time.sleep(wait_time)
                    continue
                raise
            except (APIConnectionError) as e: ##ADD MORE EXCEPTION HANDLING
                print(f"Failed to reach server. Trying again. (Attempt {attempt + 1}/3)")
                time.sleep(wait_time)
                continue

    def send_messages(self, client, messages, tools):
        while True:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages = messages,
                tools = tools
            )
            msg = response.choices[0].message
            entry = ({"role": "assistant", "content": msg.content if msg.content is not None else ""})
            if(msg.tool_calls):
                entry["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
                messages.append(entry)
                for tc in msg.tool_calls:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": self.executeTool(tc)})
                continue
            return msg.content, messages

    def get_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                "name": "get_canvas_courses",
                "description": "Retrieve all canvas courses that the user has access to",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_canvas_assignments",
                    "description": "Get a list of assignments for a specific canvas course",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "courseId": {
                                "type": "string",
                                "description": "The ID of the course in which you want to view the assignements"
                            }
                        },
                        "required": ["courseId"]
                    }
                }
            }
        ]
        return tools

    def executeTool(self, tc):
        args = json.loads(tc.function.arguments)
        func = self.AVAILABLE_FUNCTIONS.get(tc.function.name)

        if func is None:
            return f"UNKNOWN FUNCTION: {tc.function.name}"
        return json.dumps(func(args), default=str)

    def get_all_pages(self, url, params, headers):
        results = []
        while url:
            response = requests.get(url=url, params=params, headers=headers)
            response.raise_for_status()
            results.extend(response.json())

            # Find the "next" URL from the Link header, if it exists
            next_url = None
            if "Link" in response.headers:
                links = response.headers["Link"].split(",")
                for link in links:
                    if 'rel="next"' in link:
                        next_url = link.split(";")[0].strip().strip("<>")
                        break
            url = next_url

        return results


    #CANVAS TOOLS#

    def get_canvas_courses(self, canvas_token):
        url = "https://canvas.wisc.edu/api/v1/courses?enrollment_state=active&per_page=100"
        headers = {"Authorization": f"Bearer {canvas_token}"}
        raw_classes = requests.get(url=url, headers=headers).json()
        classes = []

        for course in raw_classes:
            classes.append({
                "id": course.get("id"),
                "name": course.get("name"),
            })
        return classes

    def get_canvas_assignments(self, canvas_token, courseId):
        url = f"https://canvas.wisc.edu/api/v1/courses/{courseId}/assignments?include[]=can_submit&per_page=100"
        headers = {"Authorization": f"Bearer {canvas_token}"}
        raw_assignments = self.get_all_pages(url=url, params=None, headers=headers)
        assignments = []

        for assignment in raw_assignments:
            assignments.append({
                "id": assignment.get("id"),
                "name": assignment.get("name"),
                "description": assignment.get("description"),
                "due_at": assignment.get("due_at"),
                "html_url": assignment.get("html_url"),
                "can_submit": assignment.get("can_submit"),
                "submission": assignment.get("submission")
                
            })
        return assignments

    def get_canvas_planner(self, canvas_token, courseId):
        url = "https://canvas.wisc.edu/planner/items"
        headers = {"Authorization": f"Bearer {canvas_token}"}
        raw_assignments = self.get_all_pages(url=url, params=None, headers=headers)
        assignments = []

        for assignment in raw_assignments:
            assignments.append({
                "id": assignment.get("id"),
                "name": assignment.get("name"),
                "description": assignment.get("description"),
                "due_at": assignment.get("due_at"),
                "html_url": assignment.get("html_url"),
                "can_submit": assignment.get("can_submit"),
                "submission": assignment.get("submission")
                
            })
        return assignments


    #SERVICE METHODS#

    def getStatus(self):
        return f"Agent is a Canvas Agent with the following messageLog: {self.messages}."
    
    def doTask(self, message):
        self.messages.append({"role": "system", "content": f"{message}"})
        try:
            msg, self.messages = self.try_again_loop(lambda: self.send_messages(client=self.client, messages=self.messages, tools=self.tools))
        except RuntimeError as e:
            return(f"Unable to send message: {e} Please try again.")

        return(f"canvasAgent: {msg}")

    