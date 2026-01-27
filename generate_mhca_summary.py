"""MHCA 연구 요약 PDF 생성"""

from fpdf import FPDF
import os

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # 한글 폰트 등록
        self.add_font('NanumSquare', '', '/usr/share/fonts/truetype/nanum/NanumSquare_acR.ttf')
        self.add_font('NanumSquare', 'B', '/usr/share/fonts/truetype/nanum/NanumSquare_acB.ttf')

    def header(self):
        self.set_font('NanumSquare', 'B', 14)
        self.cell(0, 10, 'ECG 부정맥 분류를 위한 MHCA 연구', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('NanumSquare', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('NanumSquare', 'B', 12)
        self.set_fill_color(230, 240, 250)
        self.cell(0, 8, title, fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(3)

    def subsection_title(self, title):
        self.set_font('NanumSquare', 'B', 10)
        self.cell(0, 6, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def body_text(self, text):
        self.set_font('NanumSquare', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def code_block(self, code):
        self.set_font('NanumSquare', '', 9)
        self.set_fill_color(245, 245, 245)
        for line in code.split('\n'):
            self.cell(0, 4.5, '  ' + line, fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(3)
        self.set_font('NanumSquare', '', 10)

    def add_table(self, headers, data, col_widths=None):
        self.set_font('NanumSquare', 'B', 9)
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_fill_color(200, 220, 240)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 6, header, border=1, fill=True, align='C')
        self.ln()

        # Data rows
        self.set_font('NanumSquare', '', 9)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 5.5, str(cell), border=1, align='C')
            self.ln()
        self.ln(3)


def main():
    pdf = PDF()
    pdf.add_page()

    # 1. 연구 개요
    pdf.section_title('1. 연구 개요')
    pdf.body_text(
        '본 연구는 ECG(심전도) 신호를 이용한 부정맥 분류에서 RR interval 정보를 '
        '효과적으로 융합하기 위해 Multi-Head Cross Attention (MHCA) 메커니즘을 적용합니다. '
        '기존 Early Fusion 방식(DAEAC)의 한계를 극복하고, ECG와 RR features 간의 '
        '관계를 학습 가능한 어텐션으로 모델링합니다.'
    )

    pdf.subsection_title('연구 목표')
    pdf.body_text(
        '- ECG 신호와 RR interval 정보의 최적 융합 방법 탐색\n'
        '- Cross-Attention을 통한 ECG-RR 상호작용 모델링\n'
        '- Early Fusion, Late Fusion, Hybrid Fusion 전략 비교\n'
        '- 부정맥 분류 성능(AUPRC, AUROC) 향상'
    )

    # 2. 데이터 구조
    pdf.section_title('2. 데이터 구조 및 실험 설정')

    pdf.subsection_title('세 가지 데이터 옵션')
    pdf.add_table(
        ['옵션', 'ECG 채널', 'RR Features', 'Fusion 전략', '설명'],
        [
            ['opt1', '1', '7차원', 'Late', 'ECG만 + Cross-Attention'],
            ['opt2', '3', '2차원', 'Early', 'DAEAC 스타일 (baseline)'],
            ['opt3', '3', '7차원', 'Early+Late', '하이브리드 융합'],
        ],
        [20, 25, 30, 35, 80]
    )

    pdf.subsection_title('RR Features 구성')
    pdf.body_text('opt1 (7차원):')
    pdf.code_block(
        '[0] pre_rr           - 현재 RR interval (ms)\n'
        '[1] post_rr          - 다음 RR interval (ms)\n'
        '[2] local_rr         - 최근 10 beat 평균 RR\n'
        '[3] pre_div_post     - RR_i / RR_{i+1}\n'
        '[4] global_rr        - 전체 평균 RR\n'
        '[5] pre_minus_global - RR_i - global_RR\n'
        '[6] pre_div_global   - RR_i / global_RR'
    )

    pdf.body_text('opt2 (2차원 - Early Fusion용):')
    pdf.code_block(
        '[0] pre_rr_ratio      - 현재 RR / 전체 평균\n'
        '[1] near_pre_rr_ratio - 현재 RR / 최근 10개 평균\n'
        '-> ECG 길이(128)만큼 repeat하여 채널로 추가'
    )

    # 3. 모델 아키텍처
    pdf.add_page()
    pdf.section_title('3. 모델 아키텍처: MACNN-SE + MHCA')

    pdf.subsection_title('전체 구조')
    pdf.code_block(
        'Input: ECG (B, 1, lead, 128)\n'
        '    |\n'
        '    v\n'
        '[Conv2d] lead 채널 감소 -> (B, 4, 1, L)\n'
        '    |\n'
        '    v\n'
        '[Stage 1] ASPP(dilation=1,6,12,18) + SE + Residual\n'
        '    |\n'
        '    v\n'
        '[Stage 2] ASPP + SE + Residual (stride=2)\n'
        '    |\n'
        '    v\n'
        '[Stage 3] BN + ReLU + ASPP + SE\n'
        '    |\n'
        '    v\n'
        '[GAP] -> ECG embedding (256)\n'
        '    |\n'
        '    v\n'
        '[MHCA Fusion] ECG(256->64) x RR(7->64) -> (128)\n'
        '    |\n'
        '    v\n'
        '[FC] -> 4 classes'
    )

    pdf.subsection_title('MHCA 모듈 상세')
    pdf.body_text(
        'Multi-Head Cross Attention Fusion은 RR features를 Query로, '
        'ECG features를 Key/Value로 사용하는 Cross-Attention 메커니즘입니다.'
    )
    pdf.code_block(
        'class MultiHeadCrossAttentionFusion:\n'
        '    RR투영:   Linear(7 -> 64) + LayerNorm + GELU\n'
        '    ECG투영:  Linear(256 -> 64)\n'
        '    \n'
        '    Cross-Attention:\n'
        '        Q = RR_proj.view(B, 1, 64)\n'
        '        K = V = ECG_proj.view(B, 1, 64)\n'
        '        Attn = MultiheadAttention(64, num_heads=1)\n'
        '    \n'
        '    FFN: Linear(64->256) -> GELU -> Linear(256->64)\n'
        '    Gate: sigmoid(alpha) * attn_output  (alpha 초기값=-2)\n'
        '    Output: concat([ECG_pool, gated_attn]) -> (128)'
    )

    pdf.subsection_title('Gate 메커니즘')
    pdf.body_text(
        'Gate는 채널별 학습 가능한 스케일링으로, RR 정보의 기여도를 제어합니다:\n'
        '- 초기값 alpha=-2.0 → sigmoid(-2) ≈ 0.12 (RR 영향 약함)\n'
        '- 학습 과정에서 자동 조정\n'
        '- 목적: 기존 ECG 성능 유지 + RR의 추가 정보 활용'
    )

    # 4. 실험 설계
    pdf.add_page()
    pdf.section_title('4. 실험 설계')

    pdf.subsection_title('Fusion 유형별 비교')
    pdf.add_table(
        ['실험명', 'fusion_type', 'lead', 'rr_dim', '데이터셋'],
        [
            ['baseline_opt2', 'None', '3', '2', 'DAEACDataset'],
            ['mhca_h1', 'mhca', '1', '7', 'ECGStandardDataset'],
            ['opt3_mhca_h1', 'mhca', '3', '7', 'Opt3Dataset'],
        ],
        [40, 30, 25, 25, 60]
    )

    pdf.subsection_title('각 실험의 특징')
    pdf.body_text(
        '1) baseline_opt2 (Early Fusion Only):\n'
        '   - DAEAC 논문 재현 실험\n'
        '   - RR ratio를 ECG와 동일 길이로 확장 후 채널 concatenation\n'
        '   - Conv2d가 3채널을 동시에 처리\n\n'
        '2) mhca_h1 (Late Fusion Only):\n'
        '   - ECG만 backbone 통과 (lead=1)\n'
        '   - RR 7차원을 Cross-Attention으로 융합\n'
        '   - RR이 ECG의 어느 부분에 주목할지 학습\n\n'
        '3) opt3_mhca_h1 (Hybrid Fusion):\n'
        '   - Early: RR ratio 2차원을 채널로 추가 (lead=3)\n'
        '   - Late: RR 7차원을 Cross-Attention으로 추가 융합\n'
        '   - 두 가지 RR 정보를 모두 활용'
    )

    # 5. 핵심 설계 결정
    pdf.section_title('5. 핵심 설계 결정사항')
    pdf.add_table(
        ['항목', '선택', '이유'],
        [
            ['Attention 방향', 'RR→ECG', 'RR이 ECG의 어느 부분에 주목할지 학습'],
            ['초기 Gate값', '-2.0', 'sigmoid(-2)≈0.12, RR 영향 약하게 시작'],
            ['Fusion 임베딩', '64차원', '표현력과 계산 효율 균형'],
            ['ASPP dilation', '(1,6,12,18)', '다중 스케일 특징 추출'],
        ],
        [45, 40, 95]
    )

    # 6. 결론
    pdf.add_page()
    pdf.section_title('6. 연구 의의 및 기대 효과')
    pdf.body_text(
        '본 연구의 MHCA 기반 ECG-RR 융합 방법론은 다음과 같은 의의를 가집니다:\n\n'
        '1) 학습 가능한 융합: 기존 단순 concatenation 대비, attention 가중치를 통해\n'
        '   ECG와 RR 간의 관계를 데이터로부터 학습합니다.\n\n'
        '2) 다양한 RR 정보 활용: 7차원의 풍부한 RR features를 활용하여\n'
        '   심박 변이도의 다양한 측면을 포착합니다.\n\n'
        '3) 유연한 융합 전략: Early, Late, Hybrid 방식을 비교하여\n'
        '   최적의 융합 지점을 탐색합니다.\n\n'
        '4) Gate 메커니즘: 학습 가능한 gate를 통해 RR 정보의 기여도를\n'
        '   자동으로 조절하여 안정적인 학습이 가능합니다.'
    )

    pdf.section_title('7. 파일 구조')
    pdf.code_block(
        'ECG_CrossAttention-stored/\n'
        '├── src/models/\n'
        '│   ├── blocks.py     # MHCA 구현 (377-503 lines)\n'
        '│   └── macnn_se.py   # 모델 구조 (16-203 lines)\n'
        '├── src/datasets/\n'
        '│   └── ecg_datasets.py  # opt1/opt2/opt3 데이터셋\n'
        '├── utils.py            # RR features 추출\n'
        '├── config.py           # 실험 설정\n'
        '└── main_autoexp.py     # 자동 실험 스크립트'
    )

    # 저장
    output_path = '/home/work/Ryuha/ECG_CrossAttention-stored/MHCA_연구요약.pdf'
    pdf.output(output_path)
    print(f'PDF saved to: {output_path}')


if __name__ == '__main__':
    main()
