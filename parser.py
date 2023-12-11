import email
import glob
import json
import mailbox
import os

import pandas as pd
from tqdm import tqdm


# Parses the enron emails.
# It extracts the body of each email and also the volume, which is the word length of a file.
def enron_parser() -> pd.DataFrame:
    # Root folder of the project
    maildir_path = os.path.expanduser("./maildir/")
    paths_array = glob.glob(f"{maildir_path}/*/_sent_mail/*")

    # Replace backslashes
    sent_mails_paths_array = [path.replace("\\", "/") for path in paths_array]

    # Store all parsed messages and the volumes
    msg_parsed = []
    msg_volumes = []

    # Open each file and parse it.
    for mail_path in tqdm(sent_mails_paths_array, total=len(sent_mails_paths_array), desc="Running Enron Parser"):
        with open(mail_path) as file_handler:
            raw_msg = email.message_from_string(file_handler.read())

            # Store the parts
            email_parts = []

            # Walk the message
            for part in raw_msg.walk():

                # If it is a text/plain content type, we retrieve the message and store inside a map.
                if part.get_content_type() == 'text/plain':
                    email_parts.append(part.get_payload())

            # Join the parts and store them
            msg_parsed.append("".join(email_parts))

            # Because we did the file_handler.read(), the cursor will be at the final position of the file when
            # using .tell(). We consider this as the volume, or the word length of a file.
            # We store the message volume.
            msg_volumes.append(file_handler.tell())

    # Create and return as dataframe
    return pd.DataFrame(data={"file_path": sent_mails_paths_array, "content": msg_parsed, "volume": msg_volumes})


# Parses the Apache mails which are stored inside .mbox files
def apache_parser(maildir_directory="./apache_ml/") -> pd.DataFrame:
    path = os.path.expanduser(maildir_directory)
    paths_array = glob.glob(f"{path}/*")

    # Replace backslashes
    mails_paths_array = [path.replace("\\", "/") for path in paths_array]

    mail_contents = []
    mail_ids = []
    msg_volumes = []

    # Loop through all .mbox files
    for mbox_path in tqdm(iterable=mails_paths_array, desc="Running Apache Parser"):

        # Loop through each mail within the .mbox file and parse it
        for mail in mailbox.mbox(mbox_path):

            # Fixes problems. If message-ID is none, skip it. Also avoid duplicates.
            if mail["Message-ID"] is None or mail["Message-ID"] in mail_ids:
                continue

            mail_content = apache_get_body(mail)
            mail_contents.append(mail_content)
            mail_volume = len(mail_content)

            # Extracts message-ID from the mail
            mail_ids.append(mail["Message-ID"])

            # Store the volume of the mail
            msg_volumes.append(mail_volume)

    # Create and return as dataframe. The name file_path is not correct, instead should be id. But this
    # helps with keeping the variable name consistent with enron parser.
    return pd.DataFrame(data={"file_path": mail_ids, "content": mail_contents, "volume": msg_volumes})


# Parses wiki dataset
def wiki_parser(maildir_directory="./wiki_plaintext/") -> pd.DataFrame:
    path = os.path.expanduser(maildir_directory)
    paths_array = glob.glob(f"{path}/*")

    # Replace backslashes
    docs_paths_array = [path.replace("\\", "/") for path in paths_array]

    # Take subset of documents
    docs = docs_paths_array[:30000]

    wiki_contents = []
    wiki_ids = []
    wiki_volumes = []

    for wiki_doc in tqdm(iterable=docs, desc="Running Wiki Parser"):
        with open(wiki_doc, "r", encoding="utf-8") as f:

            # Returns json object
            json_obj = json.load(f)

            # Maybe not necessary, but do this check just in case...
            if json_obj["id"] is None or json_obj["id"] in wiki_ids:
                continue

            # Save wiki data
            wiki_contents.append(json_obj["text"])
            wiki_ids.append(json_obj["id"])
            wiki_volumes.append(f.tell())

    return pd.DataFrame(data={"file_path": wiki_ids, "content": wiki_contents, "volume": wiki_volumes})


# Get the body of Apache msg
def apache_get_body(msg):
    parts = []
    # Walk the message
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    body = "".join(parts)

    # Notice that "to unsubscribe" is at the end of each email, so we can cut that part off.
    body = body.split("To unsubscribe")[0]
    return body