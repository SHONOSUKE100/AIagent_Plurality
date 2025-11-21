import json
import random

# 候補リスト定義
job_list = [
    "会社員", "エンジニア", "デザイナー", "教師", "看護師", "医師", "弁護士",
    "公務員", "営業職", "マーケター", "ライター", "編集者", "カメラマン",
    "美容師", "シェフ", "バリスタ", "薬剤師", "建築士", "学生", "フリーランス",
    "コンサルタント", "研究者", "プログラマー", "アーティスト", "主婦・主夫"
]

hobby_list = [
    "読書", "映画鑑賞", "音楽鑑賞", "カフェ巡り", "旅行", "料理", "ランニング",
    "ヨガ", "筋トレ", "登山", "写真撮影", "ゲーム", "アニメ", "漫画", "釣り",
    "ガーデニング", "ドライブ", "サイクリング", "散歩", "美術館巡り", "食べ歩き",
    "ショッピング", "カラオケ", "楽器演奏", "ダンス", "テニス", "ゴルフ", "サッカー",
    "野球", "バスケットボール", "水泳", "スキー", "スノーボード", "サーフィン",
    "キャンプ", "DIY", "手芸", "イラスト", "書道", "茶道"
]

location_list = [
    "東京", "横浜", "大阪", "名古屋", "札幌", "福岡", "神戸", "京都", "川崎",
    "さいたま", "広島", "仙台", "千葉", "北九州", "新潟", "浜松", "熊本",
    "相模原", "静岡", "岡山", "鹿児島", "金沢", "長野", "奈良", "松山"
]

def generate_persona():
    job = random.choice(job_list)
    age = random.randint(18, 65)
    hobbies = random.sample(hobby_list, random.randint(2, 4))
    location = random.choice(location_list)
    
    # プロフィール文のパターン
    pattern = random.randint(1, 5)
    
    if pattern == 1:
        profile = f"{location}在住の{job}。{hobbies[0]}と{hobbies[1]}が好きです。"
    elif pattern == 2:
        profile = f"{age}歳の{job}として働いています。趣味は{hobbies[0]}で、休日は{hobbies[1]}を楽しんでいます。"
    elif pattern == 3:
        hobby_text = "、".join(hobbies[:-1]) + f"、{hobbies[-1]}"
        profile = f"{location}で{job}をしています。{hobby_text}に夢中です。"
    elif pattern == 4:
        profile = f"{job}の仕事をしながら、{hobbies[0]}を楽しむ日々。{location}暮らし。"
    else:
        profile = f"{location}在住。{job}として活動中。{hobbies[0]}が趣味で、最近は{hobbies[-1]}にもハマっています。"
    
    return {
        "occupation": job,
        "age": age,
        "hobbies": hobbies,
        "residence": location,
        "user_profile": profile
    }

if __name__ == "__main__":
    persona_list = [generate_persona() for _ in range(100)]
    with open("../../data/persona/persona.json", "w", encoding="utf-8") as f:
        json.dump(persona_list, f, ensure_ascii=False, indent=4)