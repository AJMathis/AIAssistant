# IMPORTS
import os
import sqlite3
import time
import csv
import json
import numpy as np
import datetime
import requests
import keyboard
import sounddevice as sd
from collections import defaultdict
from dotenv import load_dotenv
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from openai import OpenAI, APIStatusError, APIConnectionError

import CanvasAgent

# GLOBAL VARIABLES & VARIABLE INIT
DB_PATH = "memories.db"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = WhisperModel("base", device="cpu", compute_type="int8")
speech_mode = False

agent1 = None
agent2 = None
agent3 = None
agent4 = None
agents = {1: agent1, 2: agent2, 3: agent3, 4: agent4}
agentTypes = {"canvas": CanvasAgent.CanvasAgent}

AVAILABLE_FUNCTIONS = {
    #OLD TOOL FUNCTIONS
    "retrieve_memories": lambda args: retrieve_memories(args.get("query"), args.get("top_k")),
    "save_memory": lambda args: save_memory(args.get("memory_type"), args.get("content"), args.get("importance")),
    "get_calendars": lambda args: get_calendars(),
    "get_calendar_events": lambda args: get_calendar_events(args.get("calendarId"), args.get("numDays")),
    "post_calendar_event": lambda args: post_calendar_event(args.get("calendarId"), args.get("title"), args.get("description"), args.get("start"), args.get("end"), args.get("recurrence", None)),
    "delete_calendar_event": lambda args: delete_calendar_event(args.get("calendarId"), args.get("eventId")),
    "toggle_speech_mode": lambda args: toggle_speech_mode(),
    "get_time": lambda args: get_time(),
    "get_bus_stops": lambda args: get_bus_stops(),
    "get_bus_times": lambda args: get_bus_times(args.get("depart_id"), args.get("arrive_id")),

    # NEW TOOL FUNCTIONS
    "createAgent": lambda args: createAgent(args.get("agentNum"), args.get("agentType"), args.get("message")),
    "messageAgent": lambda args: messageAgent(args.get("agentNum"), args.get("message")),
    "statusAgent": lambda args: statusAgent(args.get("agentNum"))
}
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def init_variables():
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [{"role": "system", "content": buildSystemPrompt()},{"role": "user", "content": "Hello Jarvis"}]
    tools = get_tools()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY,
    memory_type TEXT, content TEXT, importance INTEGER, embedding BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

    return api_key, client, messages, tools

# BRAIN FUNCTIONS

def agent_start():
    api_key, client, messages, tools = init_variables()
    msg, messages = try_again_loop(lambda: send_messages(client=client, messages=messages, tools=tools))
    print("\nJarvis: " + msg)
    question = input("\nYou: ")
    while question.lower() != "exit":
        messages.append({"role": "user", "content": question})
        messages.append({"role": "system", "content": f"These are the top 5 relevant memories on the subject: {retrieve_memories(question, 5)}"})
        msg, messages = send_messages(client=client, messages=messages, tools=tools)
        print("\nJarvis: " + msg)
        if(speech_mode):
            question = transcribe(push_to_talk())
            print("\nYou: " + question)
        else:
            question = input("\nYou: ")

def send_messages(client, messages, tools):
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
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": executeTool(tc)})
            continue
        return msg.content, messages

def try_again_loop(func):
    max_attempts = 4
    time_increase = 2
    for attempt in range(max_attempts):
        wait_time = 2 * attempt
        if(attempt == 3):
            print("Failed to reach server. Out of Attempts.")
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

def executeTool(tc):
    args = json.loads(tc.function.arguments)
    func = AVAILABLE_FUNCTIONS.get(tc.function.name)

    if func is None:
        return f"UNKNOWN FUNCTION: {tc.function.name}"
    return json.dumps(func(args), default=str)

def push_to_talk():
    KEY = "grave"
    SAMPLE_RATE = 16000
    FILENAME = "recording.wav"

    print(f"Hold {KEY} to talk...")
    keyboard.wait(KEY, suppress=True)

    print("\nRecording... (release to stop)")
    frames=[]

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback)
    with stream:
        while keyboard.is_pressed(KEY):
            sd.sleep(50)
    print("Recording stopped")
    audio = np.concatenate(frames, axis=0)
    write(FILENAME, SAMPLE_RATE, audio)
    return FILENAME

def transcribe(FILEPATH):
    global model

    segments, info = model.transcribe(FILEPATH, beam_size=5)
    text = "".join(segment.text for segment in segments)
    return text.strip()

#MEMORY FUNCTIONS

def buildSystemPrompt():
    with open("jarvisPrompt.txt", "r", encoding="utf-8") as f:
        return f"Today is {datetime.datetime.now()}. {f.read().strip()}"

def embed(text):
    vec = embedder.encode(text)
    return vec.astype(np.float32).tobytes()

def retrieve_memories(query, top_k):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT content, memory_type, importance, embedding FROM memories").fetchall()
    conn.close()

    if not rows:
        return []

    min_similarity = 0.35

    query_vec = embedder.encode(query)
    contents, types, importances, vecs = zip(*[(content, memory_type, importance, np.frombuffer(emb, dtype=np.float32)) for content, memory_type, importance, emb in rows])
    vecs = np.array(vecs)

    #Cosine simularity
    sims = vecs @ query_vec / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec))
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [
        {
            "content": contents[i],
            "memory_type": types[i],
            "importance": importances[i],
        }
        for i in top_indices
    ]

def save_memory(memory_type, content, importance):
    vec = embed(content)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO memories (memory_type, content, importance, embedding) VALUES (?, ?, ?, ?)", (memory_type, content, importance, vec))
    conn.commit()
    conn.close()
    print(f"Saved memory: {content}")

# NEW TOOL FUNCTIONS

def get_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_memories",
                "description": "Retrieve top memories relating to a specific query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query in which you wish to recieve relevant memories"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "How many memories you wish to recieve from most to least relevant"
                        }
                    },
                    "required": ["query", "top_k"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": "save a memory to your memory database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "description": "The type of memory being saved, such as a fact or preference"
                        },
                        "content": {
                            "type": "string",
                            "description": "The memory that is being saved."
                        },
                        "importance": {
                            "type": "integer",
                            "description": "The importance of the memory between 1-10. This is used to compare more valuable information when looking at retrieved memories"
                        }
                    },
                    "required": ["memory_type", "content", "importance"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_calendars",
                "description": "Retrieve all calendars that the user has access to",
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
                "name": "get_calendar_events",
                "description": "Retrieve all upcoming events within a calendar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendarId": {
                            "type": "string",
                            "description": "The ID of the calendar in which you want to retrieve the upcoming events"
                        },
                        "numDays": {
                            "type": "integer",
                            "description": "The number of days in which you want to recieve upcoming events \
                                for. Ex. 7 would get the events for the next week."
                        },
                    },
                    "required": ["calendarId", "numDays"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "post_calendar_event",
                "description": "Create an event within a specific calendar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendarId": {
                            "type": "string",
                            "description": "The ID of the calendar in which you want to create the event"
                        },
                        "title": {
                            "type": "string",
                            "description": "The title of the created event"
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the created event"
                        },
                        "start": {
                            "type": "string",
                            "format": "date-time",
                            "description": "The starting dateTime of the created event"
                        },
                        "end": {
                            "type": "string",
                            "format": "date-time",
                            "description": "The ending dateTime of the created event"
                        },
                        "recurrence": {
                            "type": "string",
                            "description": "An optional iCalendar RRULE string for recurring events. Examples: 'RRULE:FREQ=DAILY;COUNT=10' (repeats 10 times) or 'RRULE:FREQ=WEEKLY;UNTIL=20261231T235959Z' (repeats weekly until Dec 31, 2026)."
                        }  
                    },
                    "required": ["calendarId", "title", "description", "start", "end"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_calendar_event",
                "description": "Delete an event within a specific calendar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendarId": {
                            "type": "string",
                            "description": "The ID of the calendar in which you want to delete the event"
                        },
                        "eventId": {
                            "type": "string",
                            "description": "The ID of the event in which you want to delete"
                        }  
                    },
                    "required": ["calendarId", "eventId"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "toggle_speech_mode",
                "description": "Toggles the TTS and SST speech mode on or off",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required":[]  
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Gets the current Date and Time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required":[]  
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_bus_stops",
                "description": "Retrieves information for all bus stops in Madison, WI",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required":[]  
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_bus_times",
                "description": "Retrieves the all possible depart times, arrival times, and trip ids for departing and arriving locations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "depart_id": {
                            "type": "string",
                            "description": "The ID of the bus stop you are wishing to depart from"
                        },
                        "arrive_id": {
                            "type": "string",
                            "description": "The ID of the bus stop you are wishing to arrive at"
                        },
                    },
                    "required":["depart_id", "arrive_id"]  
                },
            }
        },

        #NEW TOOL FUNCTIONS

        {
            "type": "function",
            "function": {
                "name": "createAgent",
                "description": """Creates a Sub Agent to delegate tasks and sends its first message. Each Sub Agent type has specific tools they can utilize. "
                    You have space for 4 Agents, these can be the same or different Types. You can create an agent over an existing one to reinitialize to a new agent.
                    Returns the confirmation of Agent creation and the agent response""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agentNum": {
                            "type": "integer",
                            "description": "The agent you wish to initialize or reinitialize. Agents available are '1', '2', '3', '4'"
                        },
                        "agentType": {
                            "type": "string",
                            "description": "The Type of agent you wish to create. This determines the tools and systemPrompt it recieves. Agent types available are 'canvas'"
                        },
                        "message": {
                            "type": "string",
                            "description": """The first message you wish to send to the new Agent. The Agent will Automatically recieve a system prompt and a 'Hello, I am Jarvis' message from you.
                                This message can be a general request, an elaboration on the system prompt, or something else."""
                        },
                    },
                    "required":["agentNum", "agentType", "message"]  
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "messageAgent",
                "description": "Sends a Message to an existing initialized agent. Each Sub Agent type has specific tools they can utilize. Returns the SubAgent's Response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agentNum": {
                            "type": "integer",
                            "description": "The agent you wish to initialize or reinitialize. Agents available are '1', '2', '3', '4'"
                        },
                        "message": {
                            "type": "string",
                            "description": "The  message you wish to send to the Agent."
                        },
                    },
                    "required":["agentNum", "message"]  
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "statusAgent",
                "description": "Retreives the status of an Agent. It will retreive the agent's type and message log if it is initialized",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agentNum": {
                            "type": "integer",
                            "description": "The agent you wish to initialize or reinitialize. Agents available are '1', '2', '3', '4'"
                        }
                    },
                    "required":["agentNum"]  
                },
            }
        },
    ]
    return tools if tools != {} else None

def createAgent(agentNum, agentType, message):
    if(agentNum < 1 or agentNum > 4): return f"Invalid Agent Num: Available Agents: 1-{len(agents)}"
    agents[agentNum] = agentTypes[agentType]()
    return f"Sucessfully made agent {agentNum} a {agentType} agent! Response: {agents[agentNum].doTask(message)}"

def messageAgent(agentNum, message):
    if(agentNum < 1 or agentNum > 4): return f"Invalid Agent Num: Available Agents: 1-{len(agents)}"
    return agents[agentNum].doTask(message) if agents[agentNum] != None else "Agent is not Initialized"

def statusAgent(agentNum):
    if(agentNum < 1 or agentNum > 4): return f"Invalid Agent Num: Available Agents: 1-{len(agents)}"
    return agents[agentNum].getStatus() if agents[agentNum] != None else "Agent is not Initialized"


# OLD TOOL FUNCTIONS ------TO BE REPLACED

def fileRead(fileName):
    with open(fileName, "r", encoding="utf-8") as f:
        return f.read().strip()

def fileWrite(fileName, content):
    with open(fileName, "w", encoding="utf-8") as f:
        return f.write(content)

def get_all_pages(url, params, headers):
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

def get_all_pages_google(url, params, headers):
    results = []
    page_token = None
    while True:
        if(page_token):
            params["page_token"] = page_token

        response = requests.get(url=url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        results.extend(data.get("items",[]))

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return results

def get_access_token():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds.token

def get_calendars():
    url = "https://www.googleapis.com/calendar/v3/users/me/calendarList" 
    headers = {"Authorization": f"Bearer {get_access_token()}", "Accept": "application/json",}

    raw_calendars = requests.get(url=url, headers=headers).json().get("items", [])
    calendars = []
    for calendar in raw_calendars:
        calendars.append({
            "name": calendar.get("summary", "Unnamed Calendar"),
            "id": calendar.get("id"),
            "primary": calendar.get("primary", False)
        })
    return calendars

def get_calendar_events(calendarId, numDays):
    now = datetime.datetime.now()
    time_min = now.isoformat() + 'Z'
    time_max = (now + datetime.timedelta(days=numDays)).isoformat() + 'Z'
    
    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events"
    headers = {"Authorization": f"Bearer {get_access_token()}", "Accept": "application/json",}
    param = {"calendarId": calendarId, "timeMin": time_min, "timeMax": time_max, "singleEvents": True, "orderBy": "startTime"}

    raw_events = get_all_pages_google(url=url, params=param, headers=headers)
    events = []
    for event in raw_events:
        events.append({
            "id": event.get("id"),
            "title": event.get("summary", "No Title"), 
            "start": event["start"].get("dateTime", event["start"].get("date")),
            "end": event["end"].get("dateTime", event["end"].get("date")),
            "location": event.get("location", "Not Specified"),
            "description": event.get("description", "")
        })
    return events

def post_calendar_event(calendarId, title, description, start, end, recurrence):
    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events"
    headers = {"Authorization": f"Bearer {get_access_token()}", "Accept": "application/json"}
    body = {"calendarId": calendarId, "summary": title, "description": description, "start": {"dateTime": start, "timeZone": "America/Chicago"}, "end": {"dateTime": end, "timeZone": "America/Chicago"}}

    if recurrence:
        body["recurrence"] = [recurrence]

    requests.post(url=url, json=body, headers=headers)

    return f"Successfully added event to calendar: {calendarId}"

def delete_calendar_event(calendarId, eventId):
    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events/{eventId}"
    headers = {"Authorization": f"Bearer {get_access_token()}", "Accept": "application/json"}

    requests.delete(url=url, headers=headers)
    return f"Successfully deleted event {eventId} from calendar {calendarId}"


def toggle_speech_mode():
    global speech_mode
    speech_mode =  not speech_mode
    return f"Speech Mode is now {"on" if speech_mode else "off"}"

def get_time():
    return datetime.datetime.now()

def get_bus_stops():
    with open("stops.txt", "r", encoding="utf-8") as f:
        raw_stops = csv.DictReader(f)
        stops = []
        for stop in raw_stops:
            stops.append({
                "stop_id": stop.get("stop_id"),
                "stop_name": stop.get("stop_name"),
                "stop_desc": stop.get("stop_desc")
            })
    return stops

def get_bus_times(depart_id, arrive_id):
    trips = defaultdict(dict)
    with open("stop_times.txt", "r", encoding="utf-8") as f:
        raw_times = csv.DictReader(f)
        for time in raw_times:
            stop_id = time.get("stop_id")
            trip_id = time.get("trip_id")

            if stop_id == depart_id:
                trips[trip_id]["departure"] = {
                    "departure_time": time.get("departure_time"),
                    "stop_sequence": int(time.get("stop_sequence"))
                }
            elif stop_id == arrive_id:
                trips[trip_id]["arrival"] = {
                    "arrival_time": time.get("arrival_time"),
                    "stop_sequence": int(time.get("stop_sequence"))
                }
    results = []
    for trip_id, stops in trips.items():
        if "departure" in stops and "arrival" in stops:
            if stops["departure"]["stop_sequence"] < stops["arrival"]["stop_sequence"]:
                results.append({
                    "trip_id": trip_id,
                    "departure_time": stops["departure"]["departure_time"],
                    "arrival_time": stops["arrival"]["arrival_time"]
                })

    return results
        
# MAIN
def main():
    agent_start()

if __name__ == "__main__":
    main()