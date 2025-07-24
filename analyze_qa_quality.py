#!/usr/bin/env python3
"""
Analiza jakości datasetu Q&A przed Phase 2 fine-tuning
Sprawdza długość odpowiedzi, różnorodność i sugeruje poprawki
"""

import json
import statistics
from pathlib import Path
from collections import Counter
import argparse

def analyze_qa_dataset(dataset_file: str):
    """Analizuj jakość datasetu Q&A"""
    
    if not Path(dataset_file).exists():
        print(f"❌ Plik {dataset_file} nie istnieje")
        return
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 ANALIZA DATASETU Q&A: {dataset_file}")
    print("=" * 60)
    
    # Basic stats
    total_examples = len(data)
    questions = [item.get('instruction', '') for item in data if not item.get('input')]
    answers = [item.get('output', '') for item in data]
    
    print(f"📈 PODSTAWOWE STATYSTYKI:")
    print(f"   Łączna liczba przykładów: {total_examples}")
    print(f"   Unikalne pytania: {len(set(questions))}")
    print(f"   Duplikaty pytań: {len(questions) - len(set(questions))}")
    
    # Answer length analysis
    answer_lengths = [len(answer) for answer in answers]
    answer_word_counts = [len(answer.split()) for answer in answers]
    
    print(f"\n📏 DŁUGOŚĆ ODPOWIEDZI:")
    print(f"   Średnia długość (znaki): {statistics.mean(answer_lengths):.1f}")
    print(f"   Mediana długość (znaki): {statistics.median(answer_lengths):.1f}")
    print(f"   Najkrótsza odpowiedź: {min(answer_lengths)} znaków")
    print(f"   Najdłuższa odpowiedź: {max(answer_lengths)} znaków")
    
    print(f"\n📝 LICZBA SŁÓW:")
    print(f"   Średnia liczba słów: {statistics.mean(answer_word_counts):.1f}")
    print(f"   Mediana liczba słów: {statistics.median(answer_word_counts):.1f}")
    print(f"   Najkrótsza odpowiedź: {min(answer_word_counts)} słów")
    print(f"   Najdłuższa odpowiedź: {max(answer_word_counts)} słów")
    
    # Quality indicators
    short_answers = sum(1 for count in answer_word_counts if count < 20)
    very_short_answers = sum(1 for count in answer_word_counts if count < 10)
    good_length_answers = sum(1 for count in answer_word_counts if 20 <= count <= 100)
    
    print(f"\n🚨 OSTRZEŻENIA JAKOŚCIOWE:")
    print(f"   Bardzo krótkie odpowiedzi (<10 słów): {very_short_answers} ({very_short_answers/total_examples*100:.1f}%)")
    print(f"   Krótkie odpowiedzi (<20 słów): {short_answers} ({short_answers/total_examples*100:.1f}%)")
    print(f"   Odpowiedzi dobrej długości (20-100 słów): {good_length_answers} ({good_length_answers/total_examples*100:.1f}%)")
    
    # Show examples
    print(f"\n🔍 PRZYKŁADY:")
    
    # Shortest answer
    shortest_idx = answer_word_counts.index(min(answer_word_counts))
    print(f"\n   NAJKRÓTSZA ODPOWIEDŹ ({answer_word_counts[shortest_idx]} słów):")
    print(f"   Q: {questions[shortest_idx] if shortest_idx < len(questions) else 'N/A'}")
    print(f"   A: {answers[shortest_idx][:200]}...")
    
    # Longest answer
    longest_idx = answer_word_counts.index(max(answer_word_counts))
    print(f"\n   NAJDŁUŻSZA ODPOWIEDŹ ({answer_word_counts[longest_idx]} słów):")
    print(f"   Q: {questions[longest_idx] if longest_idx < len(questions) else 'N/A'}")
    print(f"   A: {answers[longest_idx][:200]}...")
    
    # Risk assessment
    print(f"\n⚠️  OCENA RYZYKA PHASE 2:")
    
    risk_score = 0
    risks = []
    
    if short_answers / total_examples > 0.5:
        risk_score += 3
        risks.append("🔴 WYSOKIE: >50% odpowiedzi ma <20 słów")
    elif short_answers / total_examples > 0.3:
        risk_score += 2
        risks.append("🟡 ŚREDNIE: >30% odpowiedzi ma <20 słów")
    
    if very_short_answers / total_examples > 0.2:
        risk_score += 2
        risks.append("🔴 WYSOKIE: >20% odpowiedzi ma <10 słów")
    
    if len(set(questions)) / len(questions) < 0.8:
        risk_score += 1
        risks.append("🟡 ŚREDNIE: Dużo duplikatów pytań")
    
    if total_examples < 100:
        risk_score += 1
        risks.append("🟡 ŚREDNIE: Mały dataset (<100 przykładów)")
    
    if not risks:
        print("   ✅ NISKIE RYZYKO - dataset wygląda dobrze")
    else:
        for risk in risks:
            print(f"   {risk}")
    
    # Recommendations
    print(f"\n💡 REKOMENDACJE:")
    
    if short_answers / total_examples > 0.3:
        print("   1. Przeregeneruj Q&A z wymogiem dłuższych odpowiedzi")
        print("   2. Ręcznie rozszerz najkrótsze odpowiedzi")
        print("   3. Użyj niższego learning rate w Phase 2")
    
    if total_examples < 200:
        print("   4. Wygeneruj więcej przykładów Q&A")
        print("   5. Rozważ mixing z oryginalnym korpusem")
    
    print("   6. Monitoruj długość odpowiedzi podczas treningu")
    print("   7. Użyj early stopping w Phase 2")
    print("   8. Test modelu na przykładach wymagających długich odpowiedzi")
    
    return {
        'total_examples': total_examples,
        'avg_answer_length': statistics.mean(answer_word_counts),
        'short_answer_ratio': short_answers / total_examples,
        'risk_score': risk_score
    }

def suggest_phase2_params(stats: dict):
    """Zasugeruj parametry Phase 2 na podstawie analizy"""
    
    print(f"\n🎯 SUGEROWANE PARAMETRY PHASE 2:")
    print("=" * 40)
    
    # Learning rate based on quality
    if stats['short_answer_ratio'] > 0.5:
        lr = "1e-5"
        print(f"   Learning rate: {lr} (bardzo niski - dużo krótkich odpowiedzi)")
    elif stats['short_answer_ratio'] > 0.3:
        lr = "2e-5"
        print(f"   Learning rate: {lr} (niski - trochę krótkich odpowiedzi)")
    else:
        lr = "5e-5"
        print(f"   Learning rate: {lr} (normalny)")
    
    # Epochs based on dataset size
    if stats['total_examples'] < 100:
        epochs = "1-2"
        print(f"   Epochs: {epochs} (mały dataset)")
    elif stats['total_examples'] < 500:
        epochs = "2-3"
        print(f"   Epochs: {epochs} (średni dataset)")
    else:
        epochs = "3-5"
        print(f"   Epochs: {epochs} (duży dataset)")
    
    # Other recommendations
    print(f"   LoRA rank: 16 (zachowawczy)")
    print(f"   LoRA alpha: 32")
    print(f"   Warmup steps: 10-50")
    print(f"   Save steps: 100")
    print(f"   Evaluation strategy: steps")
    print(f"   Eval steps: 100")

def main():
    parser = argparse.ArgumentParser(description="Analizuj jakość datasetu Q&A")
    parser.add_argument("dataset", help="Plik z danymi Q&A (.json)")
    parser.add_argument("--suggest-params", action="store_true", 
                       help="Zasugeruj parametry Phase 2")
    
    args = parser.parse_args()
    
    stats = analyze_qa_dataset(args.dataset)
    
    if args.suggest_params and stats:
        suggest_phase2_params(stats)

if __name__ == "__main__":
    main() 