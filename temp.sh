factors="0.9 1.0 1.1"
for factor in ${factors}; do
if [[ ${factor} != 1.0 ]]; then
echo "if"
else
echo "else"
fi
done
