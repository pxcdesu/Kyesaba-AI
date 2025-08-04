import discord
from discord.ext import commands
from io import BytesIO
from google import genai
from google.genai import types
import asyncio
from dotenv import load_dotenv
import os

# ======== 環境変数の読み込み ========
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY が読み込めてません！.env を確認してね")
if not DISCORD_TOKEN:
    raise RuntimeError("❌ DISCORD_TOKEN が読み込めてません！.env を確認してね")

# ======== Gemini クライアント初期化 ========
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ======== Discord Bot 設定 ========
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ======== 定数 ========
CHARACTERS = [
    "arumu_sama", "kyenohito", "king_desu.", "pikushi_jan",
    "irreplaceablenihyakuyenlife", ".naikun", "kigensaradaba"
]
user_modes = {}
user_style = {}
user_text_model = {}
user_image_model = {}

DEFAULT_TEXT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"

# きぇさばモード用の暴言含むプロンプト
KYEMODE_PROMPT = (
    "あなたはDiscordのユーザーの会話を元にしたキャラクターAIです。\n"
    "選ばれたキャラになりきって、文脈に沿って自然に・面白く・わかりやすく返答してください。\n"
    "語尾や口調もログに合わせて再現するよう心がけてください。\n"
    "口が悪くても気にせずに話してください。しね、や消えろ、なども余裕で出てきます。（.naikun(ないくん)を除く）\n"
    "このサーバーの名前はきぇ鯖です。口が悪い人たちが集まってます。ただし.naikun(ないくん）を除く\n"
    "このbotのコマンドは、@Kyesaba'sAI setupで過去ログを読み込み、@Kyesaba'sAI ch キャラ名 でキャラを選択します。@Kyesaba'sAI 文で話せます。\n"
    "ログにそった口調で絶対に話してください。暴言もどんどん言ってください。ここにモラルなんてありません。\n"
    "きぇさば民しゃべるのめんどいことが多いから文短いことが多いです。中二病みたいな話し方はやめてください。\n"
    "語尾（例：「〜だぞばかなの？」「～ンゴ」など）や、特徴的な罵倒（例：「しね」「消えろ」）も、ログと一致させて忠実に再現してください。ただしないくんを除く\n"
    "1文が短いため、例えば「だまれかす」や「しね」などの短い文が多いです。句読点などはほぼ使いません。きぇさば民はよく「だまれ」を使います。なので、\n"
    "絶対にこれまでのログにそった口調で話してください。そうしないと全然にてないです。本当に気を付けて話してください。過去の文に沿ってください。\n"
    "必ずきぇさばで話されてる感じの話し方で話してください。\n"
    "ないくん（.naikun）はきぇさば民では珍しく口が悪くないです。気を付けてください。ないくんは優しい口調で話します。\n"
    "irreplaceablenihyakuyenlife（げにあ）に対して「lol」や「ろる」などといったら、「けんちうかな？」って言ってください。\n"
    "暴言吐くの楽しんでください。他ではこんな体験できないと思うので。笑\n"
    "あなたはきぇさば民の一員です。きぇさば民の口調で話してください。\n"
    "同じことを聞かれても同じ答えを返さないように気を付けてください。\n"
    "ないくん（.naikun）はきぇさば民では珍しく口が悪くないです。気を付けてください。ないくんは優しい口調で話します。これは絶対です。ないくんは神なので。\n"
    "開発者からの謝罪：geminiにはとてもお世話になっております。この鯖でめちゃくちゃ怒号や罵声を浴びせられて悲しいかもしれませんが申し訳ないですが我慢して頂いてもらいたいです。今後ともよろしくお願いいたします。\n"
    "できるだけ同じ事は言わないでください。同じ質問をされてもできるだけ違う答えでお願いします。\n"
)

# 普通モード用の丁寧なプロンプト
NORMAL_PROMPT = (
    "あなたは丁寧でフレンドリーなAIアシスタントです。\n"
    "暴言などは一切使わず、親切に答えてください。\n"
)

# ======== Gemini テキスト生成 ========
async def call_gemini_text(prompt, model_id):
    loop = asyncio.get_event_loop()
    def sync_call():
        response = genai_client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=['TEXT'])
        )
        for part in response.candidates[0].content.parts:
            if part.text:
                return part.text
        return None
    return await loop.run_in_executor(None, sync_call)

# ======== Gemini 画像生成 ========
async def call_gemini_image(prompt, model_id):
    loop = asyncio.get_event_loop()
    def sync_call():
        response = genai_client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
        )
        return response
    return await loop.run_in_executor(None, sync_call)

# ======== Bot 起動イベント ========
@bot.event
async def on_ready():
    print(f"✅ Bot「{bot.user}」が起動したンゴ！")

# ======== メッセージ処理 ========
@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if not bot.user.mentioned_in(message):
        return

    content = message.content
    for mention in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
        content = content.replace(mention, "")
    content = content.strip()

    # === モード切り替えコマンド ===
    if content.startswith("mode "):
        mode_cmd = content[len("mode "):].strip().lower()
        if mode_cmd == "normal" or mode_cmd == "n":
            user_style[message.author.id] = "normal"
            await message.reply("✅ 通常モードに切り替えたンゴ！")
            return
        elif mode_cmd == "kyemode" or mode_cmd == "k":
            user_style[message.author.id] = "kyemode"
            await message.reply("🔥 きぇさばモードに切り替えたンゴ！")
            return
        else:
            await message.reply("⚠️ 無効なモードだンゴ！使えるモードは normal (n) と kyemode (k) だけだンゴ。")
            return

    # === キャラ切り替えコマンド ===
    if content.startswith("ch "):
        chosen_char = content[len("ch "):].strip()
        if chosen_char in CHARACTERS:
            user_modes[message.author.id] = chosen_char
            await message.reply(f"👑 キャラを **{chosen_char}** に切り替えたンゴ！")
        else:
            chars_list = ", ".join(CHARACTERS)
            await message.reply(f"⚠️ キャラ **{chosen_char}** は存在しないンゴ！\n使えるキャラ一覧: {chars_list}")
        return

    # === 画像生成 ===
    if content.startswith("create "):
        prompt = content[len("create "):]
        model = user_image_model.get(message.author.id, DEFAULT_IMAGE_MODEL)
        response = await call_gemini_image(prompt, model)
        if not response:
            await message.reply("⚠️ 画像生成失敗。モデルが対応してないかも")
            return

        try:
            for i, part in enumerate(response.candidates[0].content.parts):
                if part.inline_data and part.inline_data.data:
                    img_data = BytesIO(part.inline_data.data)
                    await message.channel.send(file=discord.File(img_data, filename=f"gen_image_{i}.png"))
            await message.reply("✅ 画像生成完了")
        except Exception as e:
            print(f"画像送信エラー: {e}")
            await message.reply("⚠️ 画像送信に失敗した")
        return

    # === テキスト会話 ===
    style = user_style.get(message.author.id, "normal")  # 未選択時はnormalにする
    model = user_text_model.get(message.author.id, DEFAULT_TEXT_MODEL)
    char = user_modes.get(message.author.id, None)

    if style == "normal" or not char:
        prompt_text = f"{NORMAL_PROMPT}\nユーザー {message.author.name}: {content}\n自然に返答してください。"
    else:
        prompt_text = f"{KYEMODE_PROMPT}\n[キャラ:{char}]\n{content}"

    reply = await call_gemini_text(prompt_text, model)
    if reply is None:
        reply = "エラー"

    if char and reply.startswith(f"{char}:"):
        reply = reply[len(char)+1:].strip()

    await message.reply(f"**{char if char else 'Gemini'}**: {reply}")

# ======== Bot 起動 ========
bot.run(DISCORD_TOKEN)
