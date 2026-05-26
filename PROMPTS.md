# PROMPTS LOG

## 1. 과제 목표
- 타겟 논문: DTAAD: Dual Tcn-attention networks for anomaly detection in multivariate time series data
- 목표: AI 코딩 툴을 활용하여 오픈소스 코드 구현 및 분석 수행

## 2. 사용 도구
- Claude (Anthropic)
- Google Colab
- Python (torch, pandas, numpy, scikit-learn, matplotlib)
- GitHub: https://github.com/Yu-Lingrui/DTAAD

## 3. 실제 사용한 프롬프트 요약
1. "이 논문을 구현한 오픈소스가 나와있는 링크가 https://github.com/Yu-Lingrui/DTAAD 야. 똑같이 코랩으로 여기 있는 오픈소스 불러와서 실행해보려고 하는데, 어떻게 해?"
2. "너가 실행할 수 있는 코드 짜줘"
3. "[에러 메시지 전달] 이렇게 뜨는데?" (패치 1~9 각각 반복)
4. "이 패치 내용 추가한 파일로 줘봐"
5. "[실험 결과 출력 전달] 이렇게 뜨는데 괜찮은 거야?"

## 4. 실행 과정 로그 요약
- GitHub 저장소 클론 및 패키지 설치 (pandas, tqdm, matplotlib, numpy, scikit-learn, seaborn, SciencePlots)
- Python 3.12 + PyTorch 2.x 환경 호환성 문제 9가지 발생 및 패치 적용
- 저장소 내 포함된 SMAP_MSL 데이터셋 확인 (별도 다운로드 불필요)
- preprocess.py 실행으로 데이터 정규화 및 슬라이딩 윈도우 변환 → processed/ 저장
- SMAP, MSL 데이터셋 학습 및 테스트 수행
- 20% 데이터 학습 실험 (--less 플래그) 수행
- Ablation Study 4종 (Tcn_Local, Tcn_Global, Callback, Transformer 제거) 수행

## 5. 오류와 해결
- `ModuleNotFoundError: No module named 'torchdata.datapipes'` → DGL import를 try/except로 처리 (패치 2)
- `SystemExit: 2` (argparse 충돌) → `parse_args()` → `parse_known_args()` 교체 (패치 1)
- `OSError: 'science' is not a valid package style` → plt.style.use를 try/except로 처리 (패치 3)
- `TypeError: forward() got an unexpected keyword argument 'is_causal'` → forward()에 **kwargs 추가 (패치 4)
- `AttributeError: 'DataFrame' object has no attribute 'append'` → pd.concat()으로 대체 (패치 5)
- `RuntimeError: permute() number of dimensions does not match` → dim() 체크 후 unsqueeze(0) 처리 (패치 6, 9)
- `AttributeError: object has no attribute 'transformer_encoder1'` → hasattr() 가드 추가 (패치 7)
- Testing 단계 30분 이상 무한 대기 → pot.py while True 루프를 max_iter=1000으로 제한 + fallback 임계값 추가 (패치 8)
- `FileNotFoundError: 'data/SMAP'` → 전처리 결과가 data/가 아닌 processed/ 폴더에 저장됨을 확인

## 6. 최종 산출물
- `DTAAD_Final_v4.ipynb`: 패치 1~9 모두 포함된 최종 실행 노트북
- SMAP 학습 결과: F1=0.8996, AUC=0.9892 (논문 F1=0.9023, 차이 0.003 이내)
- MSL 학습 결과: F1=0.9495, AUC=0.9916 (논문 F1=0.9495, 소수점 4자리 일치)
- SMAP 20% 학습 결과: F1=0.8863, AUC=0.9876
- Ablation Study 결과: DTAAD_Callback F1=0.8682 (논문 0.8688), DTAAD_Transformer F1=0.8682 (논문 0.8682 정확 일치)
