@echo off
sed -e "s/id /c9 /g" |^
awk "{ printf \"%%s; id %1-%%d;\n\", $0, NR }" |^
sed -e "s/;;/;/g" |^
sed -e "s/;\([^ ]\)/; \1/g"





