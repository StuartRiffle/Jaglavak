del *.ali *.o adachess.exe
gcc src\signal.c -c
gnatmake src\adachess -gnat2012 -gnatp -gnatn -O3 -f -c 
gnatbind -x adachess
gnatlink adachess signal.o
@echo "Type 'adachess' to play!"
