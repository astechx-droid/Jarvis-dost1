import asyncio
import edge_tts

async def hindi_pure():
    text = "Namaste Mr Aryan. Main Madhur hoon. Main Hindi aur English dono bashaon ko bahut achhi tarah se samajhta aur bol sakta hoon. Main aapke orders ke liye taiyaar hoon."
    voice = "hi-IN-MadhurNeural"
    filename = "jarvis_hindi_pure.mp3"
    
    print("Generating JARVIS (Pure Hindi Sample)...")
    # 1.25x speed
    communicate = edge_tts.Communicate(text, voice, rate="+25%", pitch="+0Hz")
    await communicate.save(filename)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    asyncio.run(hindi_pure())
