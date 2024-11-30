import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

def download_file(real_file_id):
    """Downloads a file
    Args:
        real_file_id: ID of the file to download
    Returns : IO object with location.
    """
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    SERVICE_ACCOUNT_FILE = 'iclr2025-7302-c75a344999ea.json'  # Download this from Google Cloud Console
    
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    try:
        # create drive api client
        service = build("drive", "v3", credentials=credentials)

        file_id = real_file_id

        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        from tqdm import tqdm
        with tqdm(total=100, desc="Downloading", unit="%") as pbar:
            while done is False:
                status, done = downloader.next_chunk()
                current = int(status.progress() * 100)
                pbar.update(current - pbar.n)
        file.seek(0)
        with open('cifar10_factor_graph.jls', 'wb') as f:
            f.write(file.read())
        return file

    except HttpError as error:
        print(f"An error occurred: {error}")
        file = None

if __name__ == "__main__":
    download_file(real_file_id="1hTv458muQd2Y1I4x0ornemQA0Ea9uZ1j")
