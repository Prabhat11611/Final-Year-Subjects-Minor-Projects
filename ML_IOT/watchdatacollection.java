import HealthKit
import Foundation

class HealthKitManager {
    static let shared = HealthKitManager() // Singleton instance
    private let healthStore: HKHealthStore
    
    // Check if HealthKit is available on the device
    private var isHealthKitAvailable: Bool {
        return HKHealthStore.isHealthDataAvailable()
    }
    
    // Health data types we want to read
    private lazy var typesToRead: Set<HKSampleType> = {
        let types: Set<HKSampleType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .respiratoryRate)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        ]
        return types
    }()
    
    // Initialize with error handling
    init() {
        guard HKHealthStore.isHealthDataAvailable() else {
            fatalError("HealthKit is not available on this device")
        }
        self.healthStore = HKHealthStore()
    }
    
    // Request authorization with error handling
    func requestAuthorization(completion: @escaping (Bool, Error?) -> Void) {
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            DispatchQueue.main.async {
                completion(success, error)
            }
        }
    }
    
    // Rest of the code remains the same...
    func fetchHealthData(startDate: Date, endDate: Date, completion: @escaping ([String: [Double]]) -> Void) {
        var allData: [String: [Double]] = [:]
        let group = DispatchGroup()
        
        // Heart Rate
        group.enter()
        fetchQuantityData(for: .heartRate, unit: HKUnit(from: "count/min"), startDate: startDate, endDate: endDate) { samples in
            allData["HeartRate"] = samples
            group.leave()
        }
        
        // Steps
        group.enter()
        fetchQuantityData(for: .stepCount, unit: HKUnit.count(), startDate: startDate, endDate: endDate) { samples in
            allData["Steps"] = samples
            group.leave()
        }
        
        // Respiratory Rate
        group.enter()
        fetchQuantityData(for: .respiratoryRate, unit: HKUnit(from: "count/min"), startDate: startDate, endDate: endDate) { samples in
            allData["RespirationRate"] = samples
            group.leave()
        }
        
        // Blood Pressure
        group.enter()
        fetchQuantityData(for: .bloodPressureSystolic, unit: HKUnit.millimeterOfMercury(), startDate: startDate, endDate: endDate) { samples in
            allData["BloodPressure"] = samples
            group.leave()
        }
        
        // Heart Rate Variability
        group.enter()
        fetchQuantityData(for: .heartRateVariabilitySDNN, unit: HKUnit.secondUnit(with: .milli), startDate: startDate, endDate: endDate) { samples in
            allData["HeartRateVariability"] = samples
            group.leave()
        }
        
        group.notify(queue: .main) {
            completion(allData)
        }
    }
    
    private func fetchQuantityData(for identifier: HKQuantityTypeIdentifier, unit: HKUnit, startDate: Date, endDate: Date, completion: @escaping ([Double]) -> Void) {
        guard let quantityType = HKObjectType.quantityType(forIdentifier: identifier) else {
            completion([])
            return
        }
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        let query = HKSampleQuery(sampleType: quantityType,
                                predicate: predicate,
                                limit: HKObjectQueryNoLimit,
                                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)]) { _, samples, error in
            
            guard let samples = samples as? [HKQuantitySample], error == nil else {
                DispatchQueue.main.async {
                    completion([])
                }
                return
            }
            
            let values = samples.map { sample in
                sample.quantity.doubleValue(for: unit)
            }
            
            DispatchQueue.main.async {
                completion(values)
            }
        }
        
        healthStore.execute(query)
    }
    
    func exportToCSV(data: [String: [Double]], fileName: String, completion: @escaping (URL?, Error?) -> Void) {
        var csvString = "Timestamp,HeartRate,Steps,RespirationRate,BloodPressure,HeartRateVariability\n"
        
        let dataLength = data["HeartRate"]?.count ?? 0
        let baseDate = Date()
        
        for i in 0..<dataLength {
            let timestamp = baseDate.addingTimeInterval(Double(i * 60))
            let heartRate = data["HeartRate"]?[i] ?? 0
            let steps = data["Steps"]?[i] ?? 0
            let respirationRate = data["RespirationRate"]?[i] ?? 0
            let bloodPressure = data["BloodPressure"]?[i] ?? 0
            let hrv = data["HeartRateVariability"]?[i] ?? 0
            
            csvString += "\(timestamp),\(heartRate),\(steps),\(respirationRate),\(bloodPressure),\(hrv)\n"
        }
        
        guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            completion(nil, NSError(domain: "HealthKit", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unable to find document directory"]))
            return
        }
        
        let fileURL = dir.appendingPathComponent("\(fileName).csv")
        
        do {
            try csvString.write(to: fileURL, atomically: true, encoding: .utf8)
            completion(fileURL, nil)
        } catch {
            completion(nil, error)
        }
    }
}