from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
import re
import json
from openai import OpenAI
import time
import os
from google.cloud import texttospeech
from pydub import AudioSegment
import subprocess
import yt_dlp
from moviepy.editor import VideoFileClip, AudioFileClip ## Make sure to insstall moviepy==1.0.3 new version doesn't work

def get_youtube_transcript(url):
    """
    Extracts the transcript from a YouTube video URL (if available) and
    writes it to transcript.json.
    
    Args:
        url (str): The full YouTube video URL.
    
    Returns:
        List[dict] or None: A list of transcript entries, each with 'text',
                            'start', and 'duration'. Returns None on failure.
    """
    # Extract the video ID from the URL
    video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not video_id_match:
        raise ValueError("Invalid YouTube URL format.")

    video_id = video_id_match.group(1)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=['ru', 'ru-RU', 'en']
        )

        # Write transcript to transcript.json
        with open('transcript.json', 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        return transcript
    
    except Exception as e:
        print(f"Transcript failed: {e}, attempting auto-caption fallback")
        try:
            auto_transcript = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['ru', 'ru-RU', 'en'],
                only_automatic_captions=True
            )

            # Write transcript to transcript.json
            with open('transcript.json', 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
                return auto_transcript
            
        except Exception as e:
            print(f"Auto-caption fallback failed: {e}")
            return None
    
def combine_segments(segments, min_duration=10):
    """
    Combine consecutive segments from the list until their span is at least min_duration.
    
    Each segment is a dict with keys "text", "start", and "duration".
    The combined segment's "start" is the start of the first segment,
    and its "duration" is defined as the start of the next group minus this start
    (for the last group, we use the end time of the final segment).
    """
    groups = []
    current_group = []
    
    # Iterate over segments to form groups
    for seg in segments:
        if not current_group:
            # Start a new group with the current segment.
            current_group.append(seg)
        else:
            group_start = current_group[0]['start']
            # Compute what the group's span would be if we added the current segment.
            seg_end = seg['start'] + seg['duration']
            potential_span = seg_end - group_start
            
            if potential_span < min_duration:
                # Not enough duration yet; add the segment to the current group.
                current_group.append(seg)
            else:
                # We have reached (or exceeded) the minimum duration.
                # Check if the current group (without this segment) already qualifies.
                last_in_group = current_group[-1]
                current_span = (last_in_group['start'] + last_in_group['duration']) - group_start
                if current_span >= min_duration:
                    # The current group already has enough duration; commit it and start a new group.
                    groups.append(current_group)
                    current_group = [seg]
                else:
                    # The current group is still too short, so include this segment and then commit.
                    current_group.append(seg)
                    groups.append(current_group)
                    current_group = []
                    
    # If any segments remain in the current group, add them as the final group.
    if current_group:
        groups.append(current_group)
    
    # Now build new segments with combined text and recalc duration.
    new_segments = []
    for i, group in enumerate(groups):
        # Combine texts with a space separator.
        combined_text = " ".join(seg['text'] for seg in group)
        group_start = group[0]['start']
        if i < len(groups) - 1:
            # Duration is the gap to the start of the next combined segment.
            next_group_start = groups[i+1][0]['start']
            new_duration = next_group_start - group_start
        else:
            # For the final group, use the end time of the last segment.
            last_seg = group[-1]
            new_duration = (last_seg['start'] + last_seg['duration']) - group_start
        
        new_segments.append({
            "text": combined_text,
            "start": round(group_start, 2),
            "duration": round(new_duration, 2)
        })
    
    return new_segments

client = OpenAI(api_key="your key here")

# Parameters
BATCH_SIZE = 20           # Initial number of new lines per batch (adjust as needed)
CONTEXT_LINES = 3         # Number of previously translated lines to include for context
MAX_ATTEMPTS = 2          # Number of attempts before reducing batch size
SLEEP_SECONDS = 1         # Delay between API calls to avoid rate limits

def load_transcript(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_transcript(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_prompt(new_lines, context_translations):
    """
    Build the initial prompt.
    """
    prompt = (
        "You are a Clash Royale translator that converts Russian text about the video game Clash Royale into English. The text is commentary of a person playing 2.6 Hog Rider\n"
        "Below you will see some context lines (already translated). DO NOT output any translation for these lines.\n"
        "After the context, you will see new lines that need translation. Translate ONLY the new lines, one per line, in the same order.\n"
        "Do not output any extra text or numbering—only the translated lines exactly, one per line.\n\n"
    )
    
    if context_translations:
        prompt += "Context (already translated, ignore these):\n"
        for idx, line in enumerate(context_translations[-CONTEXT_LINES:], start=1):
            prompt += f"{idx}. {line}\n"
        prompt += "\n"
    
    prompt += "New lines to translate:\n"
    for idx, line in enumerate(new_lines, start=1):
        prompt += f"{idx}. {line}\n"
    
    return prompt

def build_fallback_prompt(new_lines, context_translations):
    """
    Build a fallback prompt with stricter instructions.
    """
    prompt = (
        "IMPORTANT: The previous attempt did not produce exactly one translation per line.\n"
        "Please strictly follow these instructions:\n"
        "Translate the following Russian texts into English. Output exactly one translation per line corresponding to each input line, with no extra text, numbering, or explanations.\n"
    )
    
    if context_translations:
        prompt += "\nContext (already translated, ignore these):\n"
        for idx, line in enumerate(context_translations[-CONTEXT_LINES:], start=1):
            prompt += f"{idx}. {line}\n"
    
    prompt += "\nNew lines to translate:\n"
    for idx, line in enumerate(new_lines, start=1):
        prompt += f"{idx}. {line}\n"
    
    return prompt

def parse_translations(response):
    """
    Splits the API response by newline, removes any numbering and extra spaces.
    """
    lines = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # Remove leading numbering if present (e.g., "1. ")
        if line and line[0].isdigit():
            parts = line.split(".", 1)
            if len(parts) > 1:
                line = parts[1].strip()
        if line:
            lines.append(line)
    return lines

def call_translation_api(prompt, model="gpt-4o-mini-2024-07-18"):
    """
    Calls the GPT model with the prompt and returns the translated text.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates Russian text into English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error calling the API:", e)
        return None

def robust_translate_batch(new_lines, context_translations, attempt=0):
    """
    Attempts to translate a batch of lines robustly.
    
    If the response does not contain exactly one translation per line, this function:
      - Uses a fallback prompt if not already done.
      - Reduces the batch size by 1/3 and translates recursively.
      - If still failing, uses a corrective prompt.
      - If all fails, prints the erroneous response and returns empty translations.
    """
    original_batch_size = len(new_lines)
    prompt = build_prompt(new_lines, context_translations) if attempt == 0 else build_fallback_prompt(new_lines, context_translations)
    response = call_translation_api(prompt)
    if response is None:
        print("No response received for batch:", new_lines)
        return [""] * original_batch_size  # skip these lines

    parsed_lines = parse_translations(response)
    if len(parsed_lines) == original_batch_size:
        return parsed_lines
    else:
        print(f"Attempt {attempt+1}: Expected {original_batch_size} lines but got {len(parsed_lines)} lines.")
        # If we haven't reached the max attempts yet, try with the fallback prompt.
        if attempt < MAX_ATTEMPTS - 1:
            return robust_translate_batch(new_lines, context_translations, attempt=attempt+1)
        else:
            # Plan 2: Reduce batch size by 1/3 (using integer division)
            new_batch_size = max(1, (original_batch_size * 2) // 3)
            if new_batch_size < original_batch_size:
                print(f"Reducing batch size from {original_batch_size} to {new_batch_size} and splitting batch.")
                result_lines = []
                for i in range(0, original_batch_size, new_batch_size):
                    subbatch = new_lines[i:i+new_batch_size]
                    sub_result = robust_translate_batch(subbatch, context_translations)
                    result_lines.extend(sub_result)
                if len(result_lines) == original_batch_size:
                    return result_lines
            # Plan 3: Use a corrective prompt that feeds back the old response.
            corrective_prompt = (
                "Your previous response was:\n" + response + "\n\n" +
                "That output was incorrect. Please provide exactly one translation per line corresponding to the following Russian texts, with no extra text, numbering, or explanations:\n"
                + "\n".join(new_lines)
            )
            corrected_response = call_translation_api(corrective_prompt)
            corrected_lines = parse_translations(corrected_response) if corrected_response else []
            if len(corrected_lines) == original_batch_size:
                return corrected_lines
            # Final fallback: print the problematic response and skip these lines.
            print("Final error in translation response. Response received:")
            print(response)
            return [""] * original_batch_size

def translate_transcript(transcript):
    """
    Translates the transcript in batches while including context and robust error handling.
    Returns a new list of transcript dictionaries with the translated 'text'.
    """
    translated_transcript = []
    previous_translations = []  # keep track of previous translations for context
    
    # Process transcript in batches.
    i = 0
    while i < len(transcript):
        batch = transcript[i : i + BATCH_SIZE]
        new_lines = [entry['text'] for entry in batch]
        
        print(f"Translating batch starting at line {i+1} with {len(new_lines)} lines...")
        translated_lines = robust_translate_batch(new_lines, previous_translations)
        
        if len(translated_lines) != len(new_lines):
            print(f"Error: Mismatch in translation count at batch starting line {i+1}. Skipping problematic lines.")
            # In this case, we still want to keep the structure; we simply leave empty translations.
            translated_lines = [""] * len(new_lines)
        
        for j, entry in enumerate(batch):
            entry['text'] = translated_lines[j]
            translated_transcript.append(entry)
            # Update context with successful translations (even if empty)
            previous_translations.append(translated_lines[j])
        
        i += BATCH_SIZE
        time.sleep(SLEEP_SECONDS)
    
    return translated_transcript

def time_stretch_audio(input_path, output_path, speed):
    """
    Use ffmpeg's atempo filter to change audio speed without affecting pitch.
    Will only speed up (never slow down) by enforcing a minimum speed of 1.0.
    """

    if speed < 1.0:
        print(f"  ⚠️  Speed factor {speed:.3f} < 1.0 — not slowing down. Skipping stretch.")
        # Just copy original audio if slower playback would be needed
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], check=True)
        return

    # Build chain of atempo filters (each between 0.5 and 2.0)
    def build_atempo_chain(speed):
        chain = []
        while speed > 2.0:
            chain.append("atempo=2.0")
            speed /= 2.0
        chain.append(f"atempo={speed:.5f}")
        return ",".join(chain)

    filter_str = build_atempo_chain(speed)

    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", filter_str,
        output_path
    ]

    subprocess.run(command, check=True)

def generate_audio_snippets(snippets, output_dir, voice_name="en-US-Chirp3-HD-Leda"):
    """
    Generate individual TTS audio files for each snippet using the
    Google Cloud Text-to-Speech service, then post-process them
    so they do NOT exceed the specified snippet['duration'].
    Returns a list of paths to the generated files, in the same order.
    """

    os.makedirs(output_dir, exist_ok=True)
    client = texttospeech.TextToSpeechClient()
    snippet_files = []

    for i, snippet in enumerate(snippets):
        text = snippet["text"]
        desired_duration_s = snippet["duration"]

        # --- 1) Figure out TTS speaking_rate ---
        # A rough guess: ~4 seconds per 100 characters at rate=1.0
        char_count = len(text)
        estimated_time_normal_speed = (char_count / 100.0) * 4.0
        
        if desired_duration_s > 0:
            # speaking_rate = how much faster (or slower) we need to speak
            # so we don't exceed snippet's duration
            min_required_rate = (estimated_time_normal_speed / desired_duration_s) * 1.2
        else:
            min_required_rate = 1.0

        # Google TTS typically allows rates between ~0.25 to 2.0
        speaking_rate = max(1, min(2.0, min_required_rate))

        # --- 2) Synthesize the snippet with TTS ---
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )

        # --- 3) Save the snippet ---
        snippet_filename = f"snippet_{i}.mp3"
        snippet_path = os.path.join(output_dir, snippet_filename)
        with open(snippet_path, "wb") as out:
            out.write(response.audio_content)

        # --- 4) Measure actual duration & fix if needed ---
        desired_duration_ms = int(desired_duration_s * 1000)
        audio_seg = AudioSegment.from_file(snippet_path, format="mp3")
        actual_duration_ms = len(audio_seg)

        if actual_duration_ms > desired_duration_ms > 0:
            # speed_factor = how much faster we must go
            # Example: if actual=5.5s & desired=5s, ratio=5.5/5=1.1 => 1.1x faster => final ~5s
            speed_factor = actual_duration_ms / float(desired_duration_ms)
            print(f"Snippet {i} is {actual_duration_ms/1000:.2f}s, desired {desired_duration_ms/1000:.2f}s; "
                  f"speeding up by factor={speed_factor:.2f}")
                    # Use FFmpeg to do pitch-preserving time stretch
            temp_path = snippet_path.replace(".mp3", "_fixed.mp3")
            time_stretch_audio(snippet_path, temp_path, speed_factor)
            os.replace(temp_path, snippet_path)

            # Recalculate length after time-stretching
            audio_seg = AudioSegment.from_file(snippet_path, format="mp3")
            actual_duration_ms = len(audio_seg)

        snippet_files.append(snippet_path)

        print(f"Generated snippet {i}: {snippet_path}, "
              f"start={snippet['start']}, desired_dur={desired_duration_s}, "
              f"speaking_rate={speaking_rate:.2f}, final_len={actual_duration_ms/1000:.2f} s")

    return snippet_files

def stitch_snippets(snippets, snippet_files, stitched_output="stitched_output.mp3"):
    """
    Given snippet definitions + the corresponding MP3s, stitch them
    into one audio track at each snippet's `start` time.
    """
    # 1) Determine the final track length needed
    final_end_ms = 0
    snippet_segments = []

    for snippet, file_path in zip(snippets, snippet_files):
        segment = AudioSegment.from_file(file_path, format="mp3")
        snippet_length_ms = len(segment)
        snippet_start_ms = int(snippet["start"] * 1000)
        snippet_end_ms = snippet_start_ms + snippet_length_ms

        snippet_segments.append((segment, snippet_start_ms))
        if snippet_end_ms > final_end_ms:
            final_end_ms = snippet_end_ms

    # 2) Create a silent "canvas" for the final output
    final_audio = AudioSegment.silent(duration=final_end_ms)

    # 3) Overlay each snippet at the proper start
    for segment, start_ms in snippet_segments:
        final_audio = final_audio.overlay(segment, position=start_ms)

    # 4) Export the stitched track
    final_audio.export(stitched_output, format="mp3")
    print(f"Stitched audio saved to: {stitched_output}")

def download_youtube_vid(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'video.%(ext)s',  # Output file will be video.mp4 (or appropriate extension)
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading video from: {url}")
        ydl.download([url])
        print("Download complete!")

def merge_video_and_audio(video_file, audio_file, output_file):
    # Load the video file and remove its original audio
    video_clip = VideoFileClip(video_file).without_audio()
    
    # Load the external MP3 audio file
    audio_clip = AudioFileClip(audio_file)
    
    # Set the new audio to the video clip
    final_clip = video_clip.set_audio(audio_clip)
    
    # Write the result to a new file (using libx264 codec for video)
    final_clip.write_videofile(output_file, codec="libx264")

def main():
    url = input("Paste the Youtube URL: ")

    #----- Step 1: Get transcript -----

    transcript_data = get_youtube_transcript(url)
    if transcript_data:
        print("Transcript successfully retrieved and saved to transcript.json.")

    #----- Step 2: Combine segments to increase their duration -----
    with open("transcript.json", "r", encoding="utf-8") as infile:
        segments = json.load(infile)
    
    # Combine segments so that each group spans at least 10 seconds.
    combined = combine_segments(segments, min_duration=10)
    
    # Save the combined segments to an output JSON file.
    with open("combined_transcript.json", "w", encoding="utf-8") as outfile:
        json.dump(combined, outfile, indent=2, ensure_ascii=False)
    
    print("Combined segments written to combined_transcript.json")

    #----- Step 3: Translate transcript to english -----
    input_filename = "combined_transcript.json"
    output_filename = "translated_transcript.json"
    
    transcript = load_transcript(input_filename)
    print("Loaded transcript with", len(transcript), "lines.")
    
    translated_transcript = translate_transcript(transcript)
    
    save_transcript(translated_transcript, output_filename)
    print("Translation complete. Saved to", output_filename)

    #----- Step 4: Generate audio segments & combine -----
    json_file = "translated_transcript.json"
    output_dir = "tts_output"
    stitched_output_file = "final_stitched.mp3"
    voice_name = "en-US-Chirp3-HD-Leda"

    with open(json_file, "r", encoding="utf-8") as f:
        snippets = json.load(f)

    # Generate + fix snippet durations as needed
    snippet_files = generate_audio_snippets(snippets, output_dir, voice_name)

    # Stitch them together
    stitch_snippets(snippets, snippet_files, stitched_output_file)

    #----- Step 5: Download YouTube Video -----
    download_youtube_vid(url)

    #----- Step 6: Merge audio and video files together -----
    video_file = 'video.mp4'
    audio_file = 'final_stitched.mp3'
    output_file = 'final.mp4'
    merge_video_and_audio(video_file, audio_file, output_file)

if __name__ == "__main__":
    main()