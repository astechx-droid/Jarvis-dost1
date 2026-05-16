import asyncio
import edge_tts

async def madhur_hinglish_fast():
    text = "System ready. How are you sir? What can I do for you today?"
    # _to_hinglish logic would turn this into:
    # "System ready hai. Aap kaise hain sir? main kya kar sakta hoon for aap today?" (depending on complexity)
    # The actual function will handle it.
    
    hinglish_text = "System ready hai. Aap kaise hain sir? Main aapki kya madad kar sakta hoon?"
    voice = "hi-IN-MadhurNeural"
    filename = "jarvis_madhur_hinglish_fast.mp3"
    
    print("Generating JARVIS (Madhur Hinglish Fast)...")
    # 1.25x (+25%)
    communicate = edge_tts.Communicate(hinglish_text, voice, rate="+25%", pitch="+0Hz")
    await communicate.save(filename)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    asyncio.run(madhur_hinglish_fast())
