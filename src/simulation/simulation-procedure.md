用意するエージェント数
100体

エージェントの付与する情報（ペルソナ）
- occupations
- hobbies
- residence
- profile

エージェントの取れる行動
-  LIKE_POST,
-  DISLIKE_POST,
-  CREATE_POST,
-  CREATE_COMMENT,
-  LIKE_COMMENT,
-  DISLIKE_COMMENT,
-  SEARCH_POSTS,
-  SEARCH_USER,
-  TREND,
-  REFRESH,
-  DO_NOTHING,
-  FOLLOW,
-  MUTE,


env

step1:
100体のエージェントの中から二十体をランダムに選び出してpostを行わせる。

step2~
各エージェントが取れる手段から自由に選び出して、行動を行う。

