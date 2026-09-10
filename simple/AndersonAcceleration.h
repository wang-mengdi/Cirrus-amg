#pragma once

#include <Eigen/Core>
#include <deque>
#include <filesystem>
#include <memory>
#include <string>
#include <cstdint>

namespace simple {

// Type-II Anderson acceleration for a fixed-point map g=G(x). The caller must
// use the same fixed normalization of state components throughout the history.
// This module checks only algebra; CFD residual acceptance belongs to the caller.
class AndersonAcceleration {
public:
    struct Diagnostics {
        bool proposed = false;
        // "accepted" means algebraically admissible, not accepted by CFD residuals.
        bool accepted = false;
        int historySize = 0;
        int rank = 0;
        double gammaNorm = 0.0;
        double affineCoefficientNorm = 1.0;
        double residualNorm = 0.0;
        double regularization = 0.0;
        std::string reason;
    };

    explicit AndersonAcceleration(int depth = 5, double relativeRegularization = 1e-12,
                                  double gammaNormLimit = 1e6);
    ~AndersonAcceleration();
    AndersonAcceleration(const AndersonAcceleration&) = delete;
    AndersonAcceleration& operator=(const AndersonAcceleration&) = delete;
    // Fresh, solver-owned scratch directory. Files contain complete binary64
    // f/g vectors; block checksums detect truncated or damaged history reads.
    void useFileHistory(const std::filesystem::path& directory, Eigen::Index blockValues = 65536);
    bool fileHistory() const { return !historyDirectory_.empty(); }
    struct StorageStatistics {
        // Cumulative across resets. Scratch excludes caller input/output and
        // the returned candidate, and includes the temporary full residual.
        std::uint64_t writtenBytes = 0, readBytes = 0;
        Eigen::Index peakScratchValues = 0;
    };
    const StorageStatistics& storageStatistics() const { return storageStatistics_; }
    Eigen::Index residentHistoryValues() const;
    Eigen::Index storedHistoryValues() const;
    Eigen::VectorXd update(const Eigen::VectorXd& x, const Eigen::VectorXd& g);
    void reset();
    const Diagnostics& diagnostics() const { return diagnostics_; }
    bool proposed() const { return diagnostics_.proposed; }
    bool accepted() const { return diagnostics_.accepted; }
    double coefficientNorm() const { return diagnostics_.gammaNorm; }

private:
    struct FileEntry;
    struct Entry {
        // f and g store the input/output pair without a third N-vector: x=g-f.
        Eigen::VectorXd f;
        Eigen::VectorXd g;
        std::shared_ptr<FileEntry> file;
        double maximumResidual = 0.0;
    };
    int depth_;
    double relativeRegularization_;
    double gammaNormLimit_;
    Eigen::Index stateSize_ = 0;
    std::deque<Entry> history_;
    Diagnostics diagnostics_;
    std::filesystem::path historyDirectory_;
    Eigen::Index blockValues_ = 65536;
    std::uint64_t nextFile_ = 0;
    StorageStatistics storageStatistics_;
    std::shared_ptr<FileEntry> writeEntry(const Eigen::VectorXd& f, const Eigen::VectorXd& g);
    void fileGram(double scale, Eigen::MatrixXd& gram, Eigen::VectorXd& rhs);
    void fileCandidate(const Eigen::VectorXd& gamma, Eigen::VectorXd& candidate);
    void retainLatest();
};

} // namespace simple
