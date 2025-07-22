import time
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

# Loading bar like training finishing
for i in tqdm(range(100), desc="Finalizing the last neurons...", ncols=75):
    time.sleep(0.02)

time.sleep(1)

# A feel-good message with some animation
messages = [
    "✅ Model Trained",
    "🎉 Project Completed",
    "😌 Bugs Squashed",
    "📊 Results Logged",
    "🧠 Insights Gained",
    "💾 Backup Saved",
    "🌙 You did amazing today",
    "😴 Now... go get some rest 🛏️"
]

for msg in messages:
    console.print(f"[bold green]{msg}")
    time.sleep(1)

# Final motivational panel
console.print(Panel.fit(
    Text("""
╔════════════════════════════╗
║ 🎯 You Did It, Coder! 🎯   ║
║                            ║
║  Your model is now alive. ║
║  The project is complete. ║
║                            ║
║     You can sleep now.     ║
║     🌌 Goodnight 🌙        ║
╚════════════════════════════╝
""", justify="center", style="bold cyan"),
box=box.ROUNDED, title="Mission Accomplished", border_style="bright_magenta"))

