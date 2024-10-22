import Foundation
import Accelerate

// aktivasyon fonksiyonu
// girdiyi 0 ve 1 arasında normalize eder.
func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

// sigmoidin türevi
// geri yayılım sırasında kullanılır. ağırlıkların nasıl güncelleneceğini hesaplamak için gereklidir.
func sigmoidDerivative(_ x: Double) -> Double {
    return x * (1.0 - x)
}

// eğitim seti
// dört girişten oluşan bir giriş veri seti.
let inputs: [[Double]] = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

// bu sinir ağı, iki çıkış katmanına sahip.
let target1: [Double] = [0, 0, 1, 1]
let target2: [Double] = [0, 1, 0, 1]

// sinir ağı parametreleri
let inputLayerNeurons = 4 // giriş katmanı 4 nöronlu
let hiddenLayerNeurons = 4 // gizli katmanı 4 nöronlu
let outputLayerNeurons = 2 // çıkış katmanı 2 nöronlu
let learningRate = 0.1  // ödevde verilen öğrenme katsayısı
let epochs = 10000

// rastgele ağırlıklar

// gizli katman-gizli katman arasındaki bağlantıların ağırlıkları.
var inputHiddenWeights: [[Double]] = (0..<inputLayerNeurons).map { _ in
    (0..<hiddenLayerNeurons).map { _ in Double.random(in: -1...1) }
}
// gizli katman-çıkış katmanı arasındaki ağırlıklar.
var hiddenOutputWeights: [[Double]] = (0..<hiddenLayerNeurons).map { _ in
    (0..<outputLayerNeurons).map { _ in Double.random(in: -1...1) }
}

func train() {
    // epoch döngüsü sinir ağını kaç kez eğiteceğinizi belirler.
    for epoch in 0..<epochs {
        var totalError: Double = 0
        
        // eğitim setindeki her bir örnek üzerinde geri yayılımı uygula
        for (i, input) in inputs.enumerated() {
            // ileri besleme
            
            // gizli katman çıktısı
            // girişten gizli katmana: giriş katmanında gizli katmana geçerken ağırlıklarla çarpılır
            // ve toplamı alınır. sonra bu değer sigmoid fonksiyonuna sokularak gizli katman
            // nöronlarının çıkışı bulunur.
            let hiddenLayerInputs = inputHiddenWeights.map { neuronWeights in
                zip(input, neuronWeights).map(*).reduce(0, +)
            }
            let hiddenLayerOutputs = hiddenLayerInputs.map(sigmoid)
            
            // çıktı katmanı çıktısı
            // gizli katmandan çıkış katmanına: aynı şekilde gizli katmandan çıkış katmanına geçerken
            // ağırlıklar kullanarak toplam hesaplanır ve sigmoid ile sonuç çıkar.
            let outputLayerInputs = hiddenOutputWeights.map { neuronWeights in
                zip(hiddenLayerOutputs, neuronWeights).map(*).reduce(0, +)
            }
            let outputLayerOutputs = outputLayerInputs.map(sigmoid)
            
            // hedefler
            let targets = [target1[i], target2[i]]
            
            // hata hesaplama
            // çıkış katmanındaki hatalar hesaplanır. bu, ağın tahmin ettiği derğerler ile hedef
            // arasındaki farktır.
            let outputErrors = zip(targets, outputLayerOutputs).map { (target, output) in
                target - output
            }
            totalError += outputErrors.map { $0 * $0 }.reduce(0, +) / 2.0
            
            // geri yayılım (backpropagation)
            // çıktı katmanı için delta hesapla
            // çıktı katmanındaki hatalara göre ağırlık güncellemerli için deltalar hesaplanır.
            let outputDeltas = zip(outputErrors, outputLayerOutputs).map { (error, output) in
                error * sigmoidDerivative(output)
            }
            
            // gizli katman için delta hesapla
            // çıkış katmanından gelen hatalar gizli katmana geri yayılır ve bu katmandaki
            // deltalar hesaplanır.
            let hiddenErrors = hiddenOutputWeights.enumerated().map { (j, weights) in
                zip(outputDeltas, weights).map(*).reduce(0, +)
            }
            let hiddenDeltas = zip(hiddenErrors, hiddenLayerOutputs).map { (error, output) in
                error * sigmoidDerivative(output)
            }
            
            // ağırlıkları güncelle
            
            // çıkış katmanındaki ağırlıklar, delta değerleri ve öğrenme katsayısı kullanılarak güncellenir.
            for j in 0..<hiddenLayerNeurons {
                for k in 0..<outputLayerNeurons {
                    hiddenOutputWeights[j][k] += learningRate * outputDeltas[k] * hiddenLayerOutputs[j]
                }
            }
            
            // aynı şekilde, giriş katmanı ile gizli katman arasındaki ağırlıklar güncellenir.
            for i in 0..<inputLayerNeurons {
                for j in 0..<hiddenLayerNeurons {
                    inputHiddenWeights[i][j] += learningRate * hiddenDeltas[j] * input[i]
                }
            }
        }
        
        // toplam hatayı her epoch sonunda göster
        // her 1000 epoch sonunda toplam hata ekranda gösterilir.
        // hata zamanla azalmalı, bu da modelin öğreniyor olduğuna işarettir.
        if epoch % 1000 == 0 {
            print("Epoch \(epoch): Total Error: \(totalError)")
        }
    }
}

// Eğitimi başlat
train()
