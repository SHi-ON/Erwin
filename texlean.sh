# trashes the Latex generated peripheral files
# Shayan Amani - Oct 2019

find . -type f -name '*.aux' -o -name '*.gz' -o -name '*.log' -o -name '*.out' | xargs -L1 trash