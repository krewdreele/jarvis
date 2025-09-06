import os
from Assistant.calendar_service import CalendarService
from Assistant.calendar_agent import CalendarAgent
from dotenv import load_dotenv
load_dotenv('../Assistant/.env')

svc = CalendarService('../Assistant/calendar_events.json')
agent = CalendarAgent(svc)
try:
    out = agent.handle('What do I have this month?')
    print('OK:\n' + out)
except Exception as e:
    print('ERR:', type(e).__name__, str(e))
