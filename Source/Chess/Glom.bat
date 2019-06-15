copy ..\Platform.h+Operations.h+Defs.h+BitBoard.h+MoveMap.h+Position.h+MoveList.h+..\TreeSearch\ScoreCard.h+..\TreeSearch\GamePlayer.h ClKernel.cl

type Glom.h | sed -r -e "s/template<[^>]*>//g" > ClKernel.cl
copy ClPrefix.inc+ClKernel.cl+ClSuffix.inc ClKernel.h


