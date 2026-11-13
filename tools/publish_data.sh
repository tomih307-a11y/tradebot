#!/usr/bin/env bash
# usage: publish_data.sh BRANCH SRC_DIR REPO_URL   — moves finished parts from SRC_DIR into the data branch (orphan commit, history not kept)
set -e
BRANCH=$1; SRC=$2; REPO=$3
shopt -s nullglob
files=()
for f in "$SRC"/*.csv.gz; do
  if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 10 ]; then files+=("$f"); fi
done
[ ${#files[@]} -eq 0 ] && { echo "nothing to publish"; exit 0; }
rm -rf /tmp/pub
if git ls-remote --exit-code --heads "$REPO" "$BRANCH" >/dev/null 2>&1; then
  git clone -q --depth 1 -b "$BRANCH" "$REPO" /tmp/pub
else
  mkdir /tmp/pub && git -C /tmp/pub init -q
fi
cp "${files[@]}" /tmp/pub/
python -m tradebot.strategies.mtf_entry.recorder consolidate-flow /tmp/pub
python - <<'PY'
import os, time
d = "/tmp/pub"
cutoff = time.strftime("%Y%m%d", time.gmtime(time.time() - 30 * 86400))
for n in os.listdir(d):
    if n.startswith("flowday_") and n[8:16] < cutoff:
        os.remove(os.path.join(d, n))
PY
cd /tmp/pub
git config user.name "flow-bot" && git config user.email "bot@example.com"
git checkout -q --orphan snap
git add -A
git commit -q -m "flow data $(date -u +%Y-%m-%dT%H:%MZ)"
git push -q -f "$REPO" snap:"$BRANCH"
rm -f "${files[@]}"
echo "published ${#files[@]} part(s) to $BRANCH"
