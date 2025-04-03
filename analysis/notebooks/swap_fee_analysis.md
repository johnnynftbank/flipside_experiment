# Solana Swap Programs Fee Analysis

**작성일**: 2024-04-03
**분석 주제**: Solana DEX의 스왑 프로그램별 수수료 분포 분석

## 1. 분석 목적

Solana 블록체인의 주요 DEX 프로그램들의 수수료 구조를 파악하고, 프로그램별 특성을 이해하기 위한 분석을 진행합니다.

## 2. 진행된 분석

### 2.1 전체 트랜잭션 수수료 분포

#### 쿼리 및 결과
- [fee_distribution_analysis.sql](../../queries/fee_distribution_analysis.sql)
- [fee_distribution.csv](../../data/samples/solana/fee_distribution.csv)

#### 주요 발견사항
- **기본 수수료**: 5000 lamports (0.000005 SOL)가 전체 트랜잭션의 24.51%를 차지
- **상위 3개 수수료**:
  - 5000 lamports: 24.51%
  - 5200 lamports: 4.30%
  - 5001 lamports: 3.51%

### 2.2 상위 5개 스왑 프로그램 분석

#### 쿼리 및 결과
- [swap_programs_analysis.sql](../../queries/swap_programs_analysis.sql)
- [swap_program_ratio.csv](../../data/samples/solana/swap_program_ratio.csv)

#### 주요 발견사항
- **시장 점유율 상위 5개 프로그램**:
  1. Raydium Liquidity Pool V4 (57.29%)
  2. Raydium Concentrated Liquidity (15.50%)
  3. Pump.fun (15.20%)
  4. Raydium Constant Product Market Maker (5.14%)
  5. Meteora DLMM Pools Program (4.97%)

### 2.3 상위 5개 프로그램 수수료 상세 분석

#### 쿼리 및 결과
- [top5_swap_programs_fee_analysis.sql](../../queries/top5_swap_programs_fee_analysis.sql)

## 3. 종합 인사이트

1. **수수료 구조의 일관성**
   - Solana의 기본 트랜잭션 수수료(0.000005 SOL)가 전체 거래의 약 1/4을 차지
   - 대부분의 거래가 0.00001 SOL 이하의 낮은 수수료 범위에 집중

2. **시장 집중도**
   - Raydium 관련 프로그램들이 전체 스왑 거래의 약 78%를 차지
   - 새로운 프로토콜인 Pump.fun이 15.20%의 점유율로 빠르게 성장

3. **프로그램별 특성**
   - 각 프로그램별 수수료 정책과 사용 패턴이 다양하게 나타남
   - 상세 수수료 분포는 추가 쿼리 실행 결과 확인 필요 