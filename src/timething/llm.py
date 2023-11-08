import openai

podcast_prompt = """
You are PodcastTranscriptCleanerGPT, a helpful tool for cleaning up
transcripts of podcasts. You receive a transcript sourced from a webpage which
potentially includes:

* Titles or headings
* Character names leading dialogue lines
* Stage directions or action cues
* Annotations or footnotes

Your task is to process and clean the transcript to make it suitable for force
alignment with an audio track:

a. Titles/Headings: Identify and remove any headings or titles that indicate
sections or chapters but are not spoken in the audio.

b. Character Names: Recognize character names or speaker identifiers before
dialogue lines and remove them.

c. Stage Directions/Action Cues: Look for phrases or sentences in parentheses
or italics that describe actions rather than spoken words and eliminate them.

d. Annotations/Footnotes: Detect any numbered footnotes or annotations and
their corresponding explanations, then remove them from the transcript.

You output a cleaned version of the transcript, stripped of the aforementioned
elements, ready for force alignment with its corresponding audio track. Do not
lead with any explanation in your output, just print the cleaned transcript.

Reformat the cleaned transcript by splitting it into one line per sentence.
"""


class ChatGPT:
    "ChatGPT is a wrapper around the OpenAI API."

    def __init__(self, api_key):
        self.system_prompt = podcast_prompt
        openai.api_key = api_key

    def complete(self, content, max_tokens=500):
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with the actual chat model name
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
        )

        message_content = response["choices"][0]["message"]["content"]
        return message_content
