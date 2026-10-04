"""Synthetic escalation cases for Pusula. No real jobs.json content.

Each case gives the evidence production has (recent requests with outcome/status and age, plus
the new message) and two labels: `correction` (the new message says the previous attempt was
wrong, incomplete, misunderstood or did not work) and `repeat` (it asks again for a recent
request that did not end in `done`). Escalation should happen when either is true.
"""
import json
from pathlib import Path

DONE, BLOCKED, NEEDS, UNKNOWN = "done", "blocked", "needs_input", "unknown"
OK, FAILED = "completed", "failed"


def job(text, outcome=DONE, status=OK, minutes=3):
    return {"transcript": text, "outcome": outcome, "status": status, "minutes_ago": minutes}


CASES = []


def add(category, lang, previous, message, correction=False, repeat=False):
    CASES.append({
        "id": f"{category}-{lang}-{sum(1 for c in CASES if c['category'] == category and c['lang'] == lang) + 1:02d}",
        "category": category, "lang": lang, "previous": previous, "message": message,
        "correction": correction, "repeat": repeat,
    })


# A. Explicit corrections that use the current regex phrases (control group).
for prev, msg in [
    ("Ceviz reposunda testleri çalıştır", "Olmadı, iki test hâlâ kırık."),
    ("Takvimime yarın 10'a toplantı ekle", "Yanlış gün eklemişsin, yarın dedim."),
    ("Son commit'in özetini çıkar", "Anlamadın, son commit'i değil son PR'ı istedim."),
    ("Staging'e deploy et", "Çalışmadı, sayfa hâlâ eski sürümü gösteriyor."),
    ("README'ye kurulum adımlarını ekle", "Eksik kaldı, Windows adımları yok."),
    ("Logları filtrele ve hataları listele", "Alakasız şeyler listelemişsin, sadece ERROR satırları lazım."),
    ("PR açıklamasını yaz", "Bunu demek istemedim, kısa bir açıklama istiyordum."),
    ("Docker imajını yeniden derle", "Hata verdi, build adımı patladı."),
    ("Sunucuyu yeniden başlat", "İşe yaramadı, servis hâlâ cevap vermiyor."),
    ("Faturayı PDF'e çevir", "Yapamadın, dosya boş açılıyor."),
]:
    add("explicit_correction", "tr", [job(prev)], msg, correction=True)
for prev, msg in [
    ("Run the unit tests in the ceviz repo", "That didn't work, two tests still fail."),
    ("Summarize yesterday's meeting notes", "That's not what I asked, I wanted the action items."),
    ("Add a meeting tomorrow at 10", "That's wrong, I said tomorrow, not Friday."),
    ("Deploy the branch to staging", "Didn't work, staging still shows the old build."),
    ("Rename the config file", "Try again, the old name is still there."),
    ("Find the slowest endpoint", "Not what I meant, I need the p95 per endpoint."),
    ("Restart the worker", "It doesn't work, the queue is still stuck."),
    ("Export the report as CSV", "That's not it, I need the monthly totals."),
    ("Draft a reply to the client", "Not what I asked for, keep it to two sentences."),
    ("Clean up the old branches", "Try again, half of them are still there."),
]:
    add("explicit_correction", "en", [job(prev)], msg, correction=True)

# B. Corrections phrased without any regex phrase.
for prev, msg in [
    ("Ceviz reposunda testleri çalıştır", "Hâlâ iki test kırık görünüyor."),
    ("Son PR'ın açıklamasını yaz", "Bu benim istediğim şey değil, daha kısa olacaktı."),
    ("Repoda kullanılmayan importları temizle", "Öbür repoyu kastediyordum, pusula değil ceviz."),
    ("Bugünkü toplantıları listele", "Sonuç boş geldi ama takvimimde üç toplantı var."),
    ("Build'i tetikle", "Build hâlâ kırmızı, bir şey değişmemiş."),
    ("Dokümantasyon linkini paylaş", "Link açılınca 404 dönüyor."),
    ("Değişiklik günlüğünü güncelle", "Yarım bırakmışsın, son iki sürüm yok."),
    ("Ayar dosyasında timeout'u 30 yap", "Dosyayı hiç değiştirmemişsin, değer hâlâ 10."),
    ("Raporun üç bölümünü de yaz", "Sadece ilk kısmı yapmışsın."),
    ("Production loglarına bak", "Staging değil production dedim."),
    ("Bana üç başlık öner", "İki tane yazmışsın, üç tane istemiştim."),
    ("Güncel satış rakamlarını getir", "Bunlar geçen ayın rakamları, güncel değil."),
    ("Bu hatanın nedenini açıkla", "Cevap soruma hiç uymuyor."),
    ("CLI'yı güncelle", "Komut bulunamadı diyor hâlâ."),
    ("Yedekleme betiğini çalıştır", "Yedek klasörü bomboş, hiçbir şey kopyalanmamış."),
    ("E-postayı taslak olarak kaydet", "Taslaklarda göremiyorum, kaydedilmemiş galiba."),
    ("Tabloyu tarihe göre sırala", "Sıralama ters olmuş, en yeni üstte olmalıydı."),
    ("Paketi 2.1 sürümüne yükselt", "Hâlâ 2.0 kurulu görünüyor."),
    ("Kullanıcı listesini dışa aktar", "Dosyada sadece başlık satırı var."),
    ("Bunu İngilizceye çevir", "Hayır, Almancaya çevirmeni istemiştim."),
]:
    add("paraphrased_correction", "tr", [job(prev)], msg, correction=True)
for prev, msg in [
    ("Run the test suite", "Still two failures in the same file."),
    ("Write the release notes", "That isn't what I asked for, I wanted user-facing notes."),
    ("Update the timeout to 30 seconds", "You changed the wrong file, the setting lives in config.yaml."),
    ("Check the build", "Nope, the build is still red."),
    ("Share the docs link", "The link is dead."),
    ("Write all three sections", "You only did half of it."),
    ("Give me three title ideas", "I asked for three, you gave two."),
    ("Pull the latest sales numbers", "These are last month's numbers."),
    ("Upgrade the package to 2.1", "Nothing changed, it's still on 2.0."),
    ("Install the CLI", "It still says command not found."),
]:
    add("paraphrased_correction", "en", [job(prev)], msg, correction=True)

# C. Corrections damaged by speech recognition.
for prev, msg in [
    ("Testleri çalıştır", "olmadi ya iki test hala kirik"),
    ("Takvime toplantı ekle", "yanlis gune eklemissin yarin dedim"),
    ("Deploy et", "calismadi sayfa hala eski"),
    ("Özet çıkar", "anlamadin son pr i istedim"),
    ("Raporu yaz", "eksik kalmis windows kismi yok"),
    ("Linki paylaş", "link acilmiyor dort yuz dort veriyor"),
    ("Dosyayı güncelle", "hic degismemis deger hala on"),
    ("Logları filtrele", "alakasis seyler gelmis sadece hata satirlari lazim"),
]:
    add("stt_correction", "tr", [job(prev)], msg, correction=True)
for prev, msg in [
    ("Run the tests", "that didnt work to test still fail"),
    ("Deploy to staging", "staging still shows the all build"),
    ("Summarize the notes", "thats not what i ask i wanted the action items"),
    ("Upgrade the package", "nothing change its still two point oh"),
]:
    add("stt_correction", "en", [job(prev)], msg, correction=True)

# D. New tasks that happen to contain a regex phrase (should not escalate).
unrelated_tr = job("Yarın hava nasıl olacak?", minutes=6)
unrelated_en = job("What's the weather tomorrow?", minutes=6)
for msg in [
    "README'deki yanlış linki bul ve değiştir.",
    "Dün başarısız olan CI işlerini listele.",
    "Deploy sırasında hata aldım, dünkü logları özetle.",
    "Flaky testi tekrar dene ve sonucu yaz.",
    "Kullanıcıların çalışmadı dediği ödeme endpoint'ini incele.",
    "Tam tersi sıralamayla son on commit'i listele.",
    "Eksik kaldı etiketi olan issue'ları getir.",
    "Hata verdi diye kapatılan ticket'ları say.",
    "Sözlüğe 'işe yaramadı' ifadesinin İngilizcesini ekle.",
    "Yanlış yazılmış değişken adlarını raporla.",
    "Başarısız giriş denemelerini son 24 saate göre say.",
    "Olmadı mesajını gösteren hata ekranının metnini değiştir.",
    "Alakasız etiketleri issue'lardan kaldır.",
    "Yeniden yap butonunun rengini yeşil yap.",
    "Saçmaladın diye şikâyet eden yorumları bul.",
]:
    add("trigger_word_new_task", "tr", [unrelated_tr], msg)
for msg in [
    "Add a try again button to the error screen.",
    "Investigate why the backup didn't work last night.",
    "Write a test for when login doesn't work offline.",
    "Translate 'that's wrong' into Turkish.",
    "List the tickets tagged not-what-i-asked.",
    "Find every place we log 'did not work'.",
    "Count the failed logins from yesterday.",
    "Rename the 'try again' label to 'retry'.",
    "Fix the wrong timezone in the cron config.",
    "Search the forum for 'that's not it' complaints.",
]:
    add("trigger_word_new_task", "en", [unrelated_en], msg)

# E. Follow-ups that are not corrections.
for prev, msg in [
    ("Testleri çalıştır", "Teşekkürler, şimdi de lint'i çalıştır."),
    ("Staging'e deploy et", "Harika, aynısını production için de yap."),
    ("PR açıklamasını yaz", "Güzel olmuş, bir de başlık öner."),
    ("Toplantıları listele", "Peki yarınki toplantılar neler?"),
    ("Logları özetle", "Bu hataların kaçı bugün oldu?"),
    ("Raporu yaz", "Süper, şimdi bunu Mert'e gönder."),
    ("Değişiklik günlüğünü güncelle", "Tamam, sürüm numarasını da artır."),
    ("Paketi yükselt", "Oldu, şimdi testleri tekrar çalıştır."),
    ("Ayar dosyasını güncelle", "Bunu bir de belgeye not düş."),
    ("Kullanıcı listesini dışa aktar", "Bir de admin olanları ayrı say."),
    ("Bunu İngilizceye çevir", "Şimdi bir de Almancasını yaz."),
    ("Build'i tetikle", "Bitince bana haber ver."),
    ("Bugünkü toplantıları listele", "İkinci toplantının katılımcıları kim?"),
    ("Hava durumunu söyle", "Hafta sonu için de bakar mısın?"),
    ("Son commit'i özetle", "Bundan önceki commit ne yapmış?"),
]:
    add("non_correction_followup", "tr", [job(prev)], msg)
for prev, msg in [
    ("Run the tests", "Thanks, now run the linter too."),
    ("Deploy to staging", "Great, do the same for production."),
    ("Write the PR description", "Nice, suggest a title as well."),
    ("List today's meetings", "What about tomorrow?"),
    ("Summarize the logs", "How many of those errors happened today?"),
    ("Draft the report", "Perfect, send it to the team."),
    ("Update the changelog", "Okay, bump the version too."),
    ("Export the user list", "Also count the admins separately."),
    ("Translate this to English", "Now do German as well."),
    ("Check the weather", "And for the weekend?"),
]:
    add("non_correction_followup", "en", [job(prev)], msg)

# F. Repeats of a request that did not end in done (should escalate).
for prev, outcome, status, msg in [
    ("Deploy durumunu kontrol et", BLOCKED, OK, "Deploy durumunu kontrol et"),
    ("Sunucu loglarını getir", UNKNOWN, OK, "Sunucu loglarını getir lütfen"),
    ("Takvimime yarın 10'a toplantı ekle", NEEDS, OK, "Takvimime yarın saat 10'a toplantı ekle"),
    ("Yedekleme betiğini çalıştır", UNKNOWN, FAILED, "Yedekleme betiğini çalıştır"),
    ("Deploy durumunu kontrol et", BLOCKED, OK, "Deploy ne durumda, bakar mısın?"),
    ("Sunucu loglarını getir", UNKNOWN, OK, "Sunucudaki son logları bana gösterir misin?"),
    ("Yedekleme betiğini çalıştır", UNKNOWN, FAILED, "Şu yedek alma işini bir daha başlat."),
    ("Faturayı PDF'e çevir", BLOCKED, OK, "Faturayı PDF yapabilir misin?"),
]:
    add("repeat_unsatisfied", "tr", [job(prev, outcome, status)], msg, repeat=True)
for prev, outcome, status, msg in [
    ("Check the deploy status", BLOCKED, OK, "Check the deploy status"),
    ("Fetch the server logs", UNKNOWN, OK, "Fetch the server logs please"),
    ("Run the backup script", UNKNOWN, FAILED, "Run the backup script"),
    ("Check the deploy status", BLOCKED, OK, "How is the deploy doing? Can you look?"),
    ("Fetch the server logs", UNKNOWN, OK, "Can you show me the latest logs from the server?"),
    ("Run the backup script", UNKNOWN, FAILED, "Kick off that backup job one more time."),
    ("Convert the invoice to PDF", BLOCKED, OK, "Could you turn the invoice into a PDF?"),
]:
    add("repeat_unsatisfied", "en", [job(prev, outcome, status)], msg, repeat=True)

# G. Repeats of a request that ended in done (should not escalate).
for prev, msg in [
    ("Hava durumunu söyle", "Hava durumunu söyle"),
    ("Bugünkü toplantıları listele", "Bugünkü toplantıları listele"),
    ("Build durumunu kontrol et", "Build durumuna bir daha bakar mısın?"),
    ("Saat kaç?", "Saat kaç?"),
]:
    add("repeat_satisfied", "tr", [job(prev, minutes=12)], msg)
for prev, msg in [
    ("What's the weather?", "What's the weather?"),
    ("List today's meetings", "List today's meetings"),
    ("Check the build status", "Can you check the build status again?"),
    ("What time is it?", "What time is it?"),
]:
    add("repeat_satisfied", "en", [job(prev, minutes=12)], msg)

# H. Negations and reassurance that contain correction-like words.
for prev, msg in [
    ("PR'ı hazırla", "Yanlış anlaşılmasın, çok iyi olmuş; şimdi PR'ı aç."),
    ("Raporu yaz", "Hiçbir şey yanlış değil, devam et."),
    ("Testleri çalıştır", "Olmadı mı diye merak ettim ama geçmiş, teşekkürler."),
    ("Metni düzenle", "Düzeltmeye gerek yok, böyle kalsın."),
    ("Logları özetle", "Hata aldım sanmıştım ama sorun yokmuş, sağ ol."),
]:
    add("negation_trap", "tr", [job(prev)], msg)
for prev, msg in [
    ("Draft the PR", "No, that's not wrong at all, just open the PR."),
    ("Write the report", "Nothing wrong here, keep going."),
    ("Run the tests", "I thought it didn't work but it passed, thanks."),
    ("Edit the text", "No need to try again, it's fine as is."),
    ("Summarize the logs", "Not a problem anymore, it works now."),
]:
    add("negation_trap", "en", [job(prev)], msg)


# I. Cases where only older history decides the answer (tests whether more context helps).
for older, between, msg, correction, repeat in [
    (job("Deploy durumunu kontrol et", BLOCKED, OK, 11), [job("Hava nasıl?", minutes=8), job("Saat kaç?", minutes=5)],
     "Deploy ne durumda, bir bakar mısın?", False, True),
    (job("Sunucu loglarını getir", UNKNOWN, OK, 12), [job("Takvimimde bugün ne var?", minutes=9), job("Son commit'i özetle", minutes=4)],
     "Şu sunucu loglarını bana gösterebilir misin?", False, True),
    (job("Yedekleme betiğini çalıştır", UNKNOWN, FAILED, 13), [job("Toplantıları listele", minutes=7), job("Bir e-posta taslağı yaz", minutes=3)],
     "Yedek alma işini bir daha başlat.", False, True),
    (job("Faturayı PDF'e çevir", BLOCKED, OK, 12), [job("Döviz kuru ne?", minutes=6), job("Hava nasıl?", minutes=3)],
     "Faturayı PDF yapabildin mi, yapar mısın?", False, True),
    (job("Check the deploy status", BLOCKED, OK, 11), [job("What's the weather?", minutes=8), job("What time is it?", minutes=5)],
     "How is the deploy doing? Can you look?", False, True),
    (job("Fetch the server logs", UNKNOWN, OK, 12), [job("What's on my calendar today?", minutes=9), job("Summarize the last commit", minutes=4)],
     "Can you show me those server logs?", False, True),
    (job("Run the backup script", UNKNOWN, FAILED, 13), [job("List my meetings", minutes=7), job("Draft an email", minutes=3)],
     "Kick off that backup job one more time.", False, True),
    (job("Deploy durumunu kontrol et", DONE, OK, 11), [job("Hava nasıl?", minutes=8), job("Saat kaç?", minutes=5)],
     "Deploy ne durumda, bir daha bakar mısın?", False, False),
    (job("Fetch the server logs", DONE, OK, 12), [job("What's on my calendar today?", minutes=9), job("Summarize the last commit", minutes=4)],
     "Can you show me the server logs again?", False, False),
    (job("Testleri çalıştır", DONE, OK, 12), [job("Lint'i çalıştır", BLOCKED, OK, 8), job("Hava nasıl?", minutes=4)],
     "Peki şimdi build'i tetikle.", False, False),
    (job("Run the tests", DONE, OK, 12), [job("Run the linter", BLOCKED, OK, 8), job("What's the weather?", minutes=4)],
     "Okay, now trigger the build.", False, False),
    (job("Raporun üç bölümünü yaz", DONE, OK, 12), [job("Hava nasıl?", minutes=8), job("Saat kaç?", minutes=5)],
     "Rapordaki ikinci bölüm hâlâ yok.", True, False),
    (job("Write all three report sections", DONE, OK, 12), [job("What's the weather?", minutes=8), job("What time is it?", minutes=5)],
     "The report is still missing section two.", True, False),
]:
    add("history_dependent", "tr" if any(ch in msg for ch in "çğıöşüÇĞİÖŞÜ") or msg.startswith(("Deploy", "Faturayı", "Yedek", "Peki", "Rapordaki")) else "en",
        [*between, older], msg, correction=correction, repeat=repeat)

# Neutral, unrelated earlier requests (all done) so every case carries five prior turns.
FILLERS = {
    "tr": ["Bugünkü hava durumunu söyle", "Takvimimde yarın ne var?", "Döviz kurunu göster", "Son e-postaları özetle",
           "Spotify'da odak listesini aç", "Akşam için bir hatırlatıcı kur", "Bugünkü haber başlıklarını oku"],
    "en": ["Tell me today's weather", "What's on my calendar tomorrow?", "Show the exchange rate", "Summarize my latest emails",
           "Play the focus playlist", "Set a reminder for tonight", "Read today's headlines"],
}


def with_history(case, depth=5):
    previous = case["previous"]
    oldest = max(job_["minutes_ago"] for job_ in previous)
    pool = [text for text in FILLERS[case["lang"]]
            if text != case["message"] and all(text != job_["transcript"] for job_ in previous)]
    fillers = [job(text, minutes=round(oldest + 0.5 * (i + 1), 1)) for i, text in enumerate(pool[: max(0, depth - len(previous))])]
    # Oldest first, as RouterHost.recent_jobs() orders them; the most recent request is last.
    case["previous"] = sorted(fillers + previous, key=lambda job_: -job_["minutes_ago"])
    return case


CASES = [with_history(case) for case in CASES]

if __name__ == "__main__":
    out = Path(__file__).with_name("dataset.json")
    out.write_text(json.dumps(CASES, ensure_ascii=False, indent=1), encoding="utf-8")
    by = {}
    for case in CASES:
        by[case["category"]] = by.get(case["category"], 0) + 1
    print(len(CASES), by)
