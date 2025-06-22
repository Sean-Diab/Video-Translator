Video translator!

**Input:** Video in any language  
**Output:** Video with AI voice in any other language speaking at the same pace and timing as the original speaker  

## How to use:
1. In the script, specify the language you want it translated to and from
2. Run the script and paste the url of the YouTube video you want translated.
3. Wait a bit, you will see a couple files appear in the same directory. When the script is done running, the final video will be named final.mp4.

## How it works:

1. **Transcript Extraction**  
It gets the transcript of the YouTube video from the link, this comes with timestamps. It splits up those timestamps into chunks.

2. **Translation with GPT**  
It feeds multiple chunks at a time into Chat GPT with the prompt telling to to translate from language a to language b, output just the text in the chunk.

3. **Error Handling & Retries**  
- If the output returned doesn't match the exact amount of lines necessary, we try again with a different prompt
- If that doesn't work, we split the amount of chunks into the input by half
- If we still get errors, we keep splitting until there is only one chunk left
- If that still doesn't work, we leave that final line empty
- All throughout this process we keep adding more to the prompt we give Chat GPT, even showing it its previous errors.

4. **Voice Generation**  
- We use Google Cloud API Text-To-Speech to annotate the transcript. 
- We feed the individual chunks of text separately into it and then stitch them together
- If the audio for a specific segment is too slow and intrudes into the start of the next segment, we make another annotation with a faster speed. This way, we have audio that speaks exactly when the original speaker spoke.

5. **Final video creation**  
We now download the original video from YouTube, mute the original audio and overlay this audio instead. This is the final product.
