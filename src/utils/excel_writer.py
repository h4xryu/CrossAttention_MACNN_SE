"""
Excel Result Writer

실험 결과를 엑셀 템플릿에 저장하는 유틸리티 클래스입니다.

사용 예시:
    from src.utils import ExcelResultWriter

    writer = ExcelResultWriter("template.xlsx", "output.xlsx")
    writer.write_metrics("exp_name", metrics, "auprc")
    writer.write_confusion_matrix("exp_name", cm, "auprc")
"""

import shutil
import numpy as np
from openpyxl import load_workbook


class ExcelResultWriter:
    """
    엑셀 템플릿에 실험 결과를 저장하는 클래스

    Args:
        template_path: 엑셀 템플릿 파일 경로
        output_path: 결과 저장 경로
        classes: 클래스 이름 리스트 (default: ['N', 'S', 'V', 'F'])
    """

    def __init__(self, template_path: str, output_path: str, classes=None):
        self.template_path = template_path
        self.output_path = output_path
        self.classes = classes or ['N', 'S', 'V', 'F']
        self.current_row = 3  # 데이터 시작 행 (0-indexed, 헤더 3줄 이후)
        self.confusion_start_row = 0  # Confusion 시트 시작 행

        # 템플릿 복사
        shutil.copy(template_path, output_path)
        print(f"[Excel] Template copied to: {output_path}")

    def write_metrics(self, exp_name: str, metrics: dict, best_type: str):
        """
        Performance Metrics 시트에 결과 작성

        Args:
            exp_name: 실험 이름
            metrics: calculate_metrics 결과 dict
            best_type: 'auroc', 'auprc', 'recall' 등
        """
        wb = load_workbook(self.output_path)
        ws = wb['Performance Metrics']

        row = self.current_row + 1  # openpyxl은 1-indexed

        # 실험명 (best_type 포함)
        full_name = f"{exp_name}_{best_type}"
        ws.cell(row=row, column=1, value=full_name)

        # Macro metrics (columns 2-6)
        ws.cell(row=row, column=2, value=metrics.get('macro_accuracy', metrics.get('acc', 0)))
        ws.cell(row=row, column=3, value=metrics.get('macro_recall', 0))
        ws.cell(row=row, column=4, value=metrics.get('macro_specificity', 0))
        ws.cell(row=row, column=5, value=metrics.get('macro_precision', metrics.get('macro_prec', 0)))
        ws.cell(row=row, column=6, value=metrics.get('macro_f1', 0))

        # Weighted metrics (columns 7-11)
        ws.cell(row=row, column=7, value=metrics.get('weighted_accuracy', metrics.get('acc', 0)))
        ws.cell(row=row, column=8, value=metrics.get('weighted_recall', 0))
        ws.cell(row=row, column=9, value=metrics.get('weighted_specificity', 0))
        ws.cell(row=row, column=10, value=metrics.get('weighted_precision', metrics.get('weighted_prec', 0)))
        ws.cell(row=row, column=11, value=metrics.get('weighted_f1', 0))

        # Per-class metrics (N, S, V, F)
        per_class_acc = metrics.get('per_class_acc', metrics.get('per_class_accuracy', [0]*4))
        per_class_recall = metrics.get('per_class_recall', [0]*4)
        per_class_spec = metrics.get('per_class_specificity', [0]*4)
        per_class_prec = metrics.get('per_class_precision', [0]*4)
        per_class_f1 = metrics.get('per_class_f1', [0]*4)

        for i, cls in enumerate(self.classes):
            base_col = 12 + i * 5
            ws.cell(row=row, column=base_col, value=per_class_acc[i] if i < len(per_class_acc) else 0)
            ws.cell(row=row, column=base_col + 1, value=per_class_recall[i] if i < len(per_class_recall) else 0)
            ws.cell(row=row, column=base_col + 2, value=per_class_spec[i] if i < len(per_class_spec) else 0)
            ws.cell(row=row, column=base_col + 3, value=per_class_prec[i] if i < len(per_class_prec) else 0)
            ws.cell(row=row, column=base_col + 4, value=per_class_f1[i] if i < len(per_class_f1) else 0)

        wb.save(self.output_path)
        self.current_row += 1
        print(f"[Excel] Metrics written → row {row}: {full_name}")

    def write_confusion_matrix(self, exp_name: str, cm: np.ndarray, best_type: str):
        """
        Confusion 시트에 혼동 행렬 작성

        Args:
            exp_name: 실험 이름
            cm: 혼동 행렬 (4x4)
            best_type: 'auroc', 'auprc', 'recall' 등
        """
        wb = load_workbook(self.output_path)
        ws = wb['Confusion']

        # 각 블록은 8행 (EXPNAME, 헤더2줄, 데이터4줄, 빈줄1개)
        block_start = self.confusion_start_row + 1  # openpyxl은 1-indexed

        full_name = f"{exp_name}_{best_type}"

        # EXPNAME 행
        ws.cell(row=block_start, column=1, value=full_name)

        # Predicted 헤더
        ws.cell(row=block_start + 1, column=3, value="Predicted")

        # 클래스 헤더
        for i, cls in enumerate(self.classes):
            ws.cell(row=block_start + 2, column=3 + i, value=cls)

        # Actual 라벨 및 데이터
        ws.cell(row=block_start + 3, column=1, value="Actual")
        for i, cls in enumerate(self.classes):
            ws.cell(row=block_start + 3 + i, column=2, value=cls)
            for j in range(len(self.classes)):
                ws.cell(row=block_start + 3 + i, column=3 + j, value=int(cm[i, j]))

        wb.save(self.output_path)
        self.confusion_start_row += 8  # 다음 블록 위치
        print(f"[Excel] Confusion matrix written: {full_name}")

    def write_summary_row(self, exp_name: str, summary: dict):
        """
        간단한 요약 행 작성 (커스텀 컬럼)

        Args:
            exp_name: 실험 이름
            summary: {column_name: value} 형태의 dict
        """
        wb = load_workbook(self.output_path)
        ws = wb['Performance Metrics']

        row = self.current_row + 1
        ws.cell(row=row, column=1, value=exp_name)

        col = 2
        for key, value in summary.items():
            ws.cell(row=row, column=col, value=value)
            col += 1

        wb.save(self.output_path)
        self.current_row += 1
        print(f"[Excel] Summary written → row {row}: {exp_name}")
