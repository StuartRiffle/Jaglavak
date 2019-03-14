adachess: signal.o
	gnatmake src/adachess -gnat2012 -gnatp -gnatn -O3 -f -c
	#-gnatp -gnatn -O3 -f -c
	# -gnata
	gnatbind -x adachess
	gnatlink adachess signal.o

signal.o:
	gcc src/signal.c -c

clean:
	rm *.ali *.o
