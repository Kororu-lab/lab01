flowchart TD

    %% 상위 목표
    A[1. 국가 및 프로그램 조사하기]

    A1[1.1 워킹홀리데이의 이해]
    A2[1.2 원하는 국가 선택]

    A1 --> A1_1[1.1.1 기본 이해]
    A1 --> A1_2[1.1.2 국가 확인]

    A2 --> A2_1[1.2.1 목적에 맞는 국가 선택]
    A2 --> A2_2[1.2.2 자격 요건 확인(호주 예시)]

    A --> A1
    A --> A2

    B[2. 필요 서류 준비하기]

    B1[2.1 행정 서류]
    B1_1[2.1.1 여권 준비]
    B1_2[2.1.2 범죄경력 증명서 발급]
    B1_3[2.1.3 영문 운전면허증 발급]
    B1_4[2.1.4 잔고증명서 발급]

    B2[2.2 건강 관련 증명서류]
    B2_1[2.2.1 비자 신체검사 가능 병원]
    B2_2[2.2.2 병원 예약 및 검사 진행]
    B2_3[2.2.3 검진 결과 확인]

    B --> B1
    B --> B2
    B1 --> B1_1
    B1 --> B1_2
    B1 --> B1_3
    B1 --> B1_4
    B2 --> B2_1
    B2 --> B2_2
    B2 --> B2_3

    C[3. 비자 신청하기]

    C1[3.1 온라인 신청]
    C1_1[3.1.1 온라인 계정 생성]
    C1_2[3.1.2 공식 사이트 접속 및 신청서 작성]

    C2[3.2 서류 업로드 및 결제]
    C2_1[3.2.1 서류 업로드]
    C2_2[3.2.2 신청비 결제]

    C3[3.3 신청 확인]
    C3_1[3.3.1 신청 완료 이메일 확인]
    C3_2[3.3.2 추가 서류 요청 여부 확인]

    C --> C1
    C --> C2
    C --> C3
    C1 --> C1_1
    C1 --> C1_2
    C2 --> C2_1
    C2 --> C2_2
    C3 --> C3_1
    C3 --> C3_2

    D[4. 출국 준비하기]

    D1[4.1 비자 승인 및 발급 확인]
    D1_1[4.1.1 승인 여부 확인]
    D1_2[4.1.2 유효 기간 및 규정 확인]

    D2[4.2 항공권, 숙소 예약]
    D2_1[4.2.1 항공권 예약]
    D2_2[4.2.2 숙소 예약]

    D3[4.3 추가 준비]
    D3_1[4.3.1 현지 대사관 정보 및 비상 연락망 확보]
    D3_2[4.3.2 해외 보험 가입]
    D3_3[4.3.3 휴대전화 사용 계획]
    D3_4[4.3.4 승인 비자 및 서류 출력/저장]

    D --> D1
    D --> D2
    D --> D3
    D1 --> D1_1
    D1 --> D1_2
    D2 --> D2_1
    D2 --> D2_2
    D3 --> D3_1
    D3 --> D3_2
    D3 --> D3_3
    D3 --> D3_4


    %% 전체 연결 (원하는 경우 생략 가능)
    Start[시작]
    End[완료]

    Start --> A --> B --> C --> D --> End
