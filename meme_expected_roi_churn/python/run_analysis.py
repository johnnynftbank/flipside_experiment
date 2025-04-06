#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
분석 스크립트 실행을 위한 진입점 파일
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    """
    분석 스크립트 실행 함수
    """
    parser = argparse.ArgumentParser(description='밈코인 기대수익률과 이탈률 분석 실행')
    parser.add_argument('--data-path', type=str, default=None, 
                        help='분석할 데이터 파일 경로 (기본값: 샘플 데이터 사용)')
    parser.add_argument('--output-dir', type=str, default='../results',
                        help='결과 저장 디렉토리 (기본값: ../results)')
    
    args = parser.parse_args()
    
    # 실행 시작 메시지
    print("=" * 80)
    print(f"밈코인 기대수익률과 이탈률 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 현재 스크립트 디렉토리 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 결과 디렉토리 생성
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 분석 스크립트 경로
    correlations_script = os.path.join(script_dir, 'analyze_correlations.py')
    
    # 분석 스크립트 존재 확인
    if not os.path.exists(correlations_script):
        print(f"오류: 분석 스크립트를 찾을 수 없습니다: {correlations_script}")
        sys.exit(1)
    
    # 분석 스크립트 실행
    try:
        print(f"\n1. 상관관계 분석 스크립트 실행 중...")
        cmd = [sys.executable, correlations_script]
        
        # 사용자 지정 데이터 경로가 있는 경우 추가
        if args.data_path:
            cmd.extend(['--data-path', args.data_path])
        
        subprocess.run(cmd, check=True)
        print("상관관계 분석이 성공적으로 완료되었습니다.")
        
    except subprocess.CalledProcessError as e:
        print(f"오류: 분석 스크립트 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)
    
    # 실행 완료 메시지
    print("\n" + "=" * 80)
    print(f"분석이 성공적으로 완료되었습니다: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과는 다음 경로에서 확인할 수 있습니다: {os.path.abspath(output_dir)}")
    print("=" * 80)

if __name__ == "__main__":
    main() 