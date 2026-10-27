cd ~/research/grag
.venv/bin/python - <<'EOF'
import importlib.util
for m in ["nltk", "spacy", "pysbd", "bitsandbytes", "peft", "scipy"]:
    print(m, importlib.util.find_spec(m) is not None)
EOF
ls /home/yaokang/research/testimony_rag/models/
ls ~/research/grag/results_llama/musique | head -60
df -h ~ | tail -1; free -g | head -2; nproc
