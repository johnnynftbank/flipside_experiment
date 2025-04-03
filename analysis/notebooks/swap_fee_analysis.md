# Solana Swap Programs Fee Analysis

**작성일**: 2024-04-03
**분석 주제**: Solana DEX의 스왑 프로그램별 수수료 분포 분석
**분석 기간**: 최근 30일 (2024-03-04 ~ 2024-04-02)

## 1. 분석 목적

Solana 블록체인의 주요 DEX 프로그램들의 수수료 구조를 파악하고, 프로그램별 특성을 이해하기 위한 분석을 진행합니다.

## 2. 데이터 개요

- **분석 대상**: Solana 메인넷의 트랜잭션 및 스왑 데이터
- **데이터 기간**: 
  - 트랜잭션 수수료 분석: 최근 7일
  - 스왑 프로그램 분석: 최근 30일
- **데이터 규모**:
  - 전체 스왑 건수: 약 6억 건
  - 고유 사용자 수: 약 640만 명

## 3. 진행된 분석

### 3.1 전체 트랜잭션 수수료 분포

#### 쿼리 및 결과
- [fee_distribution_analysis.sql](../../queries/fee_distribution_analysis.sql)
- [fee_distribution.csv](../../data/samples/solana/fee_distribution.csv)

#### 주요 발견사항
- **분석 기간**: 최근 7일
- **데이터 규모**: 상위 20개 수수료 유형, 각 유형별 10건 이상의 트랜잭션
- **기본 수수료**: 5000 lamports (0.000005 SOL)가 전체 트랜잭션의 24.51% 차지 (약 1.45억 건)
- **상위 3개 수수료**:
  - 5000 lamports: 24.51% (145,148,933건)
  - 5200 lamports: 4.30% (25,439,398건)
  - 5001 lamports: 3.51% (20,795,287건)

### 3.2 상위 5개 스왑 프로그램 분석

#### 쿼리 및 결과
- [swap_programs_analysis.sql](../../queries/swap_programs_analysis.sql)
- [swap_program_ratio.csv](../../data/samples/solana/swap_program_ratio.csv)

#### 주요 발견사항
- **분석 기간**: 최근 30일 (2024-03-04 ~ 2024-04-02)
- **전체 스왑 건수**: 599,350,920건
- **시장 점유율 상위 5개 프로그램**:
  1. Raydium Liquidity Pool V4: 57.29% (343,427,073건)
  2. Raydium Concentrated Liquidity: 15.50% (92,920,369건)
  3. Pump.fun: 15.20% (91,116,412건)
  4. Raydium Constant Product Market Maker: 5.14% (30,834,804건)
  5. Meteora DLMM Pools Program: 4.97% (29,812,009건)

### 3.3 상위 5개 프로그램 수수료 상세 분석

#### 쿼리 및 결과
- [top5_swap_programs_fee_analysis.sql](../../queries/top5_swap_programs_fee_analysis.sql)

## 4. 종합 인사이트

1. **수수료 구조의 일관성**
   - Solana의 기본 트랜잭션 수수료(0.000005 SOL)가 전체 거래의 약 1/4을 차지
   - 대부분의 거래가 0.00001 SOL 이하의 낮은 수수료 범위에 집중
   - 분석된 전체 트랜잭션 중 약 1.45억 건이 기본 수수료로 처리됨

2. **시장 집중도**
   - Raydium 관련 프로그램들이 전체 스왑 거래의 약 78% 차지 (467,182,246건)
   - 새로운 프로토콜인 Pump.fun이 15.20%의 점유율(91,116,412건)로 빠르게 성장
   - 상위 5개 프로그램이 전체 스왑의 98.1%를 처리

3. **프로그램별 특성**
   - 각 프로그램별 수수료 정책과 사용 패턴이 다양하게 나타남
   - 모든 주요 프로그램이 최근 30일 동안 지속적으로 활성화된 상태 유지
   - 프로그램별 고유 사용자 수는 Raydium V4가 가장 많은 641만 명으로 확인 