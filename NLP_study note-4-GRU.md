# 복습     

​     

# GRU, Gated Recurrent Unit    

​    

뉴욕대학교 조경현 교수님이 집필한 논문에서 제안      

LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄였다!

<br/>

## GRU      

LSTM은 출력, 입력, 삭제 게이트 3개의 게이트 존재     

__GRU는 업데이트 게이트와 리셋 게이트 두 가지 게이트만 존재__     

![image](https://wikidocs.net/images/page/22889/GRU.PNG)

​     

반드시 LSTM 대신 GRU를 사용하는 것이 좋지는 않다!     

데이터 양이 적을 때는, 매개 변수의 양이 적은 GRU가 조금 더 낫고      

데이터 양이 더 많으면 LSTM이 더 낫다고 알려져 있다!

<br/>

```python
# 실제 GRU 은닉층을 추가하는 코드.
model.add(GRU(hidden_size, input_shape=(timesteps, input_dim)))
```

