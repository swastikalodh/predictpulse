"""utils/recs.py — personalised recommendation engine"""

def get_recommendations(score: float) -> list:
    if score < 0.35:
        return [
            {"icon":"🌟","title":"Keep Shining",    "desc":"Your wellness looks excellent. Maintain sleep and exercise habits."},
            {"icon":"🏃","title":"Active Breaks",   "desc":"10-min movement breaks every 90 mins boost focus by 23%."},
            {"icon":"📖","title":"Creative Reading","desc":"15 min of fiction daily reduces cortisol by up to 68%."},
            {"icon":"🎵","title":"Music Therapy",   "desc":"Lo-fi or classical music during study enhances retention."},
        ]
    if score < 0.65:
        return [
            {"icon":"⏰","title":"Pomodoro Method", "desc":"25 min work + 5 min rest. Protect your recovery windows."},
            {"icon":"😴","title":"Sleep Hygiene",   "desc":"Aim 7–8 hours. No screens 1h before bed. Same wake time."},
            {"icon":"🫁","title":"Box Breathing",   "desc":"4s inhale, 4s hold, 4s exhale, 4s hold — calms anxiety fast."},
            {"icon":"📝","title":"Priority Matrix", "desc":"Eisenhower: focus on important, not just urgent tasks."},
            {"icon":"🤝","title":"Study Circles",   "desc":"Peer collaboration reduces isolation and builds accountability."},
        ]
    return [
        {"icon":"🚨","title":"Seek Counseling",  "desc":"Please speak with your institution's counselor immediately."},
        {"icon":"🛑","title":"Mandatory Rest",   "desc":"Take 24–48 hours completely offline. Your brain needs reset."},
        {"icon":"💬","title":"Talk It Out",      "desc":"Share with a trusted person. Vulnerability is strength."},
        {"icon":"🌙","title":"Sleep Priority",   "desc":"8+ hours non-negotiable. Everything else can wait."},
        {"icon":"📵","title":"Digital Detox",    "desc":"Social media < 30 min/day. Protect your mental bandwidth."},
    ]
