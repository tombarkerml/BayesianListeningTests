from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json

# If modifying these scopes, delete the file token.pickle.
#SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEETS_READ_WRITE_SCOPE = 'https://www.googleapis.com/auth/spreadsheets'
SCOPES = [SHEETS_READ_WRITE_SCOPE]

# The ID and range of a sample spreadsheet.
#SAMPLE_SPREADSHEET_ID = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms' #google sample sheet
SAMPLE_SPREADSHEET_ID = '1JGRd59WRZ7nr6cgwKJz_IKicDfZutj2UtcjHkC5IOBk'
#SAMPLE_RANGE_NAME = 'Sheet1!A2:E'

class Sheets_IO(object):

    def __init__(self, SPREADSHEET_ID):

        self.service = self.connect_to_sheets()
        self.spreadsheet_id = SPREADSHEET_ID


    def connect_to_sheets(self):
        """Shows basic usage of the Sheets API.
        Prints values from a sample spreadsheet.
        """
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'creds/credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return build('sheets', 'v4', credentials=creds)



    def append_to_sheet(self, data = [['a1'], ['a2'], ['a3']], sheet_name='Sheet1'):


        self.service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheet_id,
            range= sheet_name + "!A:Z",
            body={
                "majorDimension": "ROWS",
                "values": data
            },
            valueInputOption="USER_ENTERED"
        ).execute()
