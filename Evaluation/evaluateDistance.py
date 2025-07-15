import evaluate
import numpy as np
import pandas as pd
import re


skip_sim = True

if __name__ == '__main__':
    if not skip_sim:
        Rouge = evaluate.load('rouge')
        Meteor = evaluate.load('meteor')
        Bleu = evaluate.load('bleu')

    pattern_location = re.compile(r'.*<location>(.*)</location>', re.DOTALL)
    pattern_comment = re.compile(r'.*<comment>(.*)</comment>', re.DOTALL)

    def process_short_(text):
        target = []
        for t in text:
            t_ = t.strip()
            if len(t_) > 3:
                target.append(t_)
        return target

    def process_position(text):
        text_lines = text.splitlines()
        text_lines_processed = []
        for t in text_lines:
            if t.strip() == "":
                continue
            elif t.startswith("```"):
                continue
            if t.startswith("+") or t.startswith("-"):
                t = t[1:].strip()
            text_lines_processed.append(t)
        text_lines_processed = process_short_(text_lines_processed)
        return text_lines_processed


    def compute_union(a, b, c, value_error_count):
        try:
            all_indices = []
            for elem in b + c:
                reversed_idx = a[::-1].index(elem)
                original_idx = len(a) - 1 - reversed_idx
                all_indices.append(original_idx)
            min_idx = min(all_indices)
            max_idx = max(all_indices)
            return a[min_idx: max_idx + 1], value_error_count, False
        except ValueError as e:
            value_error_count += 1
            return a, value_error_count, True

    def calculate_(completion, target_position, target_review, diff):
        position_match = pattern_location.search(completion)
        comment_match = pattern_comment.search(completion)

        if not position_match:
            position_score = 0.0
            GIoU = -1.0
        else:
            position = position_match.group(1)
            union_diff, value_error_count, value_error = compute_union(diff, position, target_position, 0)

            c_len = len(union_diff)
            position_text_ = process_position(position)
            position_target = process_position(target_position)
            # diff = process_position(diff)
            # for p in position_text_:
            #     if p not in diff:
            #         position_text_.remove(p)
            position_text_ = set(position_text_)
            position_target = set(position_target)
            if len(position_text_ | position_target) == 0:
                position_score = 0.0
                GIoU = -1.0
            else:
                position_score = len(position_text_ & position_target) / len(position_text_ | position_target)
                GIoU = len(position_text_ & position_target) / len(position_text_ | position_target) - 1 + len(position_text_ | position_target) / c_len


        if skip_sim or not comment_match:
            meteor = 0
            rouge = 0
            bleu = 0
        else:
            review = comment_match.group(1)

            try:
                bleu = Bleu.compute(predictions=[review],
                                    references=[[target_review]])["bleu"]
            except ZeroDivisionError:
                bleu = 0.0
            rouge = Rouge.compute(predictions=[review],
                                  references=[target_review],
                                  rouge_types=['rouge1'],
                                  use_aggregator=True)['rouge1']
            meteor = Meteor.compute(predictions=[review],
                                    references=[target_review])["meteor"]

        return position_score, GIoU, bleu, rouge, meteor

    df = pd.read_parquet("Data/ReviewCommentFileHere.parquet")

    target_lines = [
        # line name here
        ]


    for target_line in target_lines:
        IoUs = []
        GIoUs = []
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        for i, row in df.iterrows():
            completion = row[target_line]
            target_position = row["review_position"]
            target_review = row["review_comment"]
            diff = row["diff"]

            IoU, GIoU, bleu, rouge, meteor = calculate_(completion, target_position, target_review, diff)
            IoUs.append(IoU)
            GIoUs.append(GIoU)
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            meteor_scores.append(meteor)

        print(f"File: {target_line}, IoU: {np.mean(IoUs)}, GIoU: {np.mean(GIoUs)}, Bleu: {sum(bleu_scores) / len(bleu_scores)}, Rouge: {sum(rouge_scores) / len(rouge_scores)}, Meteor: {sum(meteor_scores) / len(meteor_scores)}")