# Git Github 이 대체 뭐지...



## 1. Git

**Git**? 	__버전관리__를 위한 프로그램 (Version Control System)

ex) 엊그제 회의 내용을 '회의록0321.txt'에 저장

​      어제     회의 내용을 '회의록0322.txt'에 저장

​      오늘     회의 내용을 '회의록0323.txt'에 저장

​	-> Git 은 여러 파일을 생성할 필요없이 **'회의록.txt'** 하나의 파이렝 버전으로 남기고 싶은 내용을 __commit__ 하면 된다!



뿐만 아니라 Git 을 활용하면 __협업이 가능__하다

Github 같은 Remote respository 에 소스 코드를 올려, 본인의 소스 코드와 동료의 소스 코드를 하나의 소스 코드로 합칠 수 있으며, 각각 독립적으로 버전을 관리할 수 있다. 



## 2. Git 주요 용어 및 구성

**1) 영역**

- working directory
  - 현재 작업하고 있는 공간
  - Git 이 관리하고 있지만, 아직 추적(track) 하고 있지 않은 상태
- index
  - stage 또는 staging area 라고 하며, 준비 공간
  - Git 이 추적하고 있으며, 버전으로 등록되기 전 상태
- respository
  - 저장소를 의미
    - local repository : 본인 pc에 존재하는 저장소
    - remote repository : Github, Gitlab 같은 원격 저장소

![compose](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile23.uf.tistory.com%2Fimage%2F99518A335A1B63FB10F32F)

**2) Flow **

 * git init
    * .git 폴더 생성
    * .git 폴더가 있어야 파일 추적 가능 -> Git 과 관련된 작업 가능
 * git add
    * working directory의 변경된 작업 파일을 staging area 로 추가
 * git commit
    * staging area 의 내용을 local reposiory 에 확정 
 * git push
    * local repository 의 내용을 remote repository 로 업로드
 * git pull
    * local repository 의 내용을 remote repository 에서 가져오기
 * git clone
    *  .git 을 포함한 remote repository 의 파일들을 local repository 에 복사
    * Github 에서 zip 파일로 받으면 .git 폴더가 없다는 것이 명령어와의 차이점

![flow](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile22.uf.tistory.com%2Fimage%2F99A19D335A1B65750A6006)

**3) 협업 - 병합 **

 + git branch
   	+ 독립된 working directory 를 의미
      	+ 브랜치를 통해 프로젝트 참여자마다 브랜치를 가져서 독립된 작업 공간을 갖는다
      	+ 테스트 및 백업 등의 용도로 사용할 수도 있다
+ head
  + 포인터를 의미 
  + 지금 작업하고 있는 branch를 가르킨다
+ merge
  + 2개의 branch 에서 작업한 다른 내용을 하나로 합치는 것
  + 현재 branch를 기준으로 병합됨
  + 만약 두 branch 가 같은 파일의 같은 곳을 수정했다면, 충돌 (merge conflict)이 발생 이를 해결해야함
    + 해당 이슈 관계자들이 상의하여 수동으로 충돌 해결
    + 충돌이 발생하지 않으려면 작업 내용이 겹치지 않도록 분리시키는 것이 좋다



![merge](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile10.uf.tistory.com%2Fimage%2F998DD4335A1B6745220862)

< 기존에는 master 브랜치만 있는데 branch를 생성해서 새로운 working directory가 생성되었고, commit을 한 후 master로 병합( merge )하는 것>

----------

  

## 이제 깃허브 시작해보자!

참고: https://geundung.dev/46    



### 이것만 기억해도 충분한 것 같다 나중에는!

git add *
git status
git commit -m "메모하고싶은 내용 따옴표안에 적기" 
git push origin master





#### 1. 깃(Git) 설치!

https://gitforwindows.org/

설치가 완료되면 바탕화면이나 폴더 안에서 우클릭해보세요

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile28.uf.tistory.com%2Fimage%2F999C034C5AEA84DA3CB612" width = "400px">

위와같이 Git GUI Here, Git Bash Here가 추가되어있으면 정상적으로 설치된 것!



#### 2.Github 에 저장소 만들기

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile9.uf.tistory.com%2Fimage%2F99AB44425AEA8A1F22276C" width = "400px">

저장소는 정말 간단하게 생성할 수 있다! 

기본값 Public   /   Private는  유료 서비스,,,

README.md  <- 여기서 생성하지 않고 따로 직접 생성할것이기 때문에 __체크 X__



**Create repository** 버튼을 누르면

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile29.uf.tistory.com%2Fimage%2F99AF2F425AEA8A202D187B" width = "600px">

https://github.com/이름/저장소.git  <- 이 부분을 **복사** !!!



#### 3.원격저장소를 컴퓨터로 가져오기

윈도우의 경우 가져올 폴더에서 마우스 우클릭 -> **Git Bash Here** 을 선택 !

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile21.uf.tistory.com%2Fimage%2F99C9A3425AEA8A20214436" width = "600px">



##### 깃을 처음 설치하셨으면 아래와 같이 설정 

git config --global user.name "여러분 깃허브 이름"

git config --global user.email "여러분 이메일"

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile5.uf.tistory.com%2Fimage%2F9962D7425AEA8A2025F632" width = "600px">





아까 복사한 저장소 주소를 이용해

###### git clone **복사한 저장소주소**

명령어를 입력해주시면 위 사진과 같이 저장소 이름과 같은 폴더가 생긴다! 



<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile26.uf.tistory.com%2Fimage%2F9938634A5AEA8A212485DA" width = "600px">

