#include "AndersonAcceleration.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <system_error>

namespace {
// Accidental corruption check, not a cryptographic authentication mechanism.
std::uint64_t historyChecksum(const double* values, Eigen::Index count) {
    std::uint64_t hash=14695981039346656037ull;
    const auto* bytes=reinterpret_cast<const unsigned char*>(values);
    for(std::size_t i=0;i<std::size_t(count)*sizeof(double);++i) {hash^=bytes[i];hash*=1099511628211ull;}
    return hash;
}
struct BlockSum {
    double value=0.,tail=0.;
    void add(double x) {
        const double sum=value+x;
        tail+=std::abs(value)>=std::abs(x)?(value-sum)+x:(x-sum)+value;
        value=sum;
    }
    double result() const {return value+tail;}
};
}

namespace simple {

struct AndersonAcceleration::FileEntry {
    std::filesystem::path path;
    Eigen::Index values=0,block=0;
    std::vector<std::uint64_t> fChecks,gChecks;
    bool owned=false;
    ~FileEntry() {if(owned){std::error_code error;std::filesystem::remove(path,error);}}
    std::ifstream open(bool g) const {
        if(std::filesystem::file_size(path)!=2*std::uintmax_t(values)*sizeof(double))
            throw std::runtime_error("Anderson history file length changed");
        std::ifstream input(path,std::ios::binary);
        if(!input)throw std::runtime_error("Cannot read Anderson history");
        if(g)input.seekg(std::streamoff(values)*sizeof(double));
        if(!input)throw std::runtime_error("Cannot seek Anderson history");
        return input;
    }
    void read(std::ifstream& input,bool g,Eigen::Index index,double* data,Eigen::Index count) const {
        input.read(reinterpret_cast<char*>(data),std::streamsize(count*sizeof(double)));
        if(!input||input.gcount()!=std::streamsize(count*sizeof(double)))
            throw std::runtime_error("Incomplete Anderson history block");
        const auto& checks=g?gChecks:fChecks;
        if(index<0||std::size_t(index)>=checks.size()||historyChecksum(data,count)!=checks[std::size_t(index)])
            throw std::runtime_error("Anderson history checksum mismatch");
    }
};

AndersonAcceleration::AndersonAcceleration(int depth, double relativeRegularization,
                                           double gammaNormLimit)
    : depth_(depth), relativeRegularization_(relativeRegularization), gammaNormLimit_(gammaNormLimit) {
    if (depth < 0 || depth > 32 || !(relativeRegularization >= 0.0)
        || !std::isfinite(relativeRegularization) || !(gammaNormLimit > 0.0)
        || !std::isfinite(gammaNormLimit))
        throw std::invalid_argument("Invalid Anderson history, regularization, or coefficient limit");
}

AndersonAcceleration::~AndersonAcceleration() {
    history_.clear();
    if(fileHistory()) {std::error_code error;std::filesystem::remove(historyDirectory_,error);}
}

void AndersonAcceleration::useFileHistory(const std::filesystem::path& directory,Eigen::Index blockValues) {
    if(stateSize_||!history_.empty()||fileHistory()||directory.empty()||blockValues<1||blockValues>1048576)
        throw std::invalid_argument("Invalid or late Anderson file-history configuration");
    const auto path=std::filesystem::absolute(directory).lexically_normal();
    if(!std::filesystem::create_directories(path))throw std::runtime_error("Anderson history directory must be fresh");
    historyDirectory_=path;blockValues_=blockValues;
}

Eigen::Index AndersonAcceleration::residentHistoryValues() const {
    Eigen::Index count=0;for(const auto& entry:history_)count+=entry.f.size()+entry.g.size();return count;
}
Eigen::Index AndersonAcceleration::storedHistoryValues() const {
    return Eigen::Index(history_.size())*2*stateSize_;
}

std::shared_ptr<AndersonAcceleration::FileEntry> AndersonAcceleration::writeEntry(const Eigen::VectorXd& f,const Eigen::VectorXd& g) {
    auto entry=std::make_shared<FileEntry>();entry->path=historyDirectory_/("entry_"+std::to_string(nextFile_++)+".bin");
    entry->values=f.size();entry->block=blockValues_;
    if(std::filesystem::exists(entry->path))throw std::runtime_error("Anderson history file already exists");
    std::ofstream output(entry->path,std::ios::binary);
    if(!output)throw std::runtime_error("Cannot create Anderson history");entry->owned=true;
    for(bool second:{false,true}) {
        const auto& values=second?g:f;auto& checks=second?entry->gChecks:entry->fChecks;
        for(Eigen::Index first=0;first<values.size();first+=blockValues_) {
            const Eigen::Index count=std::min(blockValues_,values.size()-first);
            checks.push_back(historyChecksum(values.data()+first,count));
            output.write(reinterpret_cast<const char*>(values.data()+first),std::streamsize(count*sizeof(double)));
            if(!output)throw std::runtime_error("Cannot write Anderson history block");
            storageStatistics_.writtenBytes+=std::uint64_t(count)*sizeof(double);
        }
    }
    output.flush();if(!output)throw std::runtime_error("Cannot flush Anderson history");
    output.close();if(output.fail())throw std::runtime_error("Cannot close Anderson history");
    return entry;
}

void AndersonAcceleration::fileGram(double scale,Eigen::MatrixXd& gram,Eigen::VectorXd& rhs) {
    const int entries=int(history_.size()),columns=entries-1;
    std::vector<std::ifstream> streams;streams.reserve(entries);
    for(const auto& entry:history_)streams.push_back(entry.file->open(false));
    std::vector<BlockSum> sums(std::size_t(columns)*columns),right(columns);
    for(Eigen::Index first=0,block=0;first<stateSize_;first+=blockValues_,++block) {
        const Eigen::Index count=std::min(blockValues_,stateSize_-first);
        Eigen::MatrixXd values(count,entries);Eigen::VectorXd differenceI(count),differenceJ(count);
        storageStatistics_.peakScratchValues=std::max(storageStatistics_.peakScratchValues,count*(entries+2));
        for(int j=0;j<entries;++j) {
            history_[j].file->read(streams[j],false,block,values.col(j).data(),count);
            storageStatistics_.readBytes+=std::uint64_t(count)*sizeof(double);
        }
        for(int i=0;i<columns;++i) {
            differenceI=values.col(i+1)/scale-values.col(i)/scale;
            right[i].add(differenceI.dot(values.col(entries-1)/scale));
            for(int j=0;j<=i;++j) {
                if(j==i)sums[std::size_t(i)*columns+j].add(differenceI.squaredNorm());
                else {
                    differenceJ=values.col(j+1)/scale-values.col(j)/scale;
                    sums[std::size_t(i)*columns+j].add(differenceI.dot(differenceJ));
                }
            }
        }
    }
    for(int i=0;i<columns;++i) {
        rhs[i]=right[i].result();
        for(int j=0;j<=i;++j)gram(j,i)=gram(i,j)=sums[std::size_t(i)*columns+j].result();
    }
}

void AndersonAcceleration::fileCandidate(const Eigen::VectorXd& gamma,Eigen::VectorXd& candidate) {
    const int entries=int(history_.size());std::vector<std::ifstream> streams;streams.reserve(entries);
    for(const auto& entry:history_)streams.push_back(entry.file->open(true));
    for(Eigen::Index first=0,block=0;first<stateSize_;first+=blockValues_,++block) {
        const Eigen::Index count=std::min(blockValues_,stateSize_-first);Eigen::MatrixXd values(count,entries);
        storageStatistics_.peakScratchValues=std::max(storageStatistics_.peakScratchValues,count*entries);
        for(int j=0;j<entries;++j) {
            history_[j].file->read(streams[j],true,block,values.col(j).data(),count);
            storageStatistics_.readBytes+=std::uint64_t(count)*sizeof(double);
        }
        for(int i=0;i<entries-1;++i)candidate.segment(first,count).noalias()-=gamma[i]*(values.col(i+1)-values.col(i));
    }
}

void AndersonAcceleration::reset() {
    history_.clear();
    stateSize_ = 0;
    diagnostics_ = Diagnostics{};
}

void AndersonAcceleration::retainLatest() {
    while (history_.size() > 1) history_.pop_front();
}

Eigen::VectorXd AndersonAcceleration::update(const Eigen::VectorXd& x, const Eigen::VectorXd& g) {
    diagnostics_ = Diagnostics{};
    if (x.size() == 0 || x.size() != g.size() || !x.allFinite() || !g.allFinite())
        throw std::invalid_argument("Anderson requires finite nonempty input/output vectors of matching size");
    if (stateSize_ != 0 && stateSize_ != x.size())
        throw std::invalid_argument("Anderson state size changed; reset its history first");
    stateSize_ = x.size();
    Entry entry;
    entry.f = g - x;
    if (!fileHistory()) entry.g = g;
    if (!entry.f.allFinite()) throw std::invalid_argument("Anderson fixed-point residual overflowed");
    diagnostics_.residualNorm = entry.f.stableNorm();
    if (!std::isfinite(diagnostics_.residualNorm))
        throw std::invalid_argument("Anderson fixed-point residual norm overflowed");
    if (depth_ == 0) {
        diagnostics_.reason = "disabled";
        return g;
    }
    if (fileHistory()) {
        entry.maximumResidual = entry.f.cwiseAbs().maxCoeff();
        storageStatistics_.peakScratchValues = std::max(storageStatistics_.peakScratchValues, stateSize_);
        entry.file = writeEntry(entry.f, g);
        entry.f.resize(0);
    }
    history_.push_back(std::move(entry));
    while (history_.size() > static_cast<std::size_t>(depth_ + 1)) history_.pop_front();
    diagnostics_.historySize = static_cast<int>(history_.size());
    const int columns = static_cast<int>(history_.size()) - 1;
    if (columns == 0) {
        diagnostics_.reason = "history warmup";
        return g;
    }
    if (diagnostics_.residualNorm == 0.0) {
        diagnostics_.reason = "exact fixed point";
        retainLatest();
        return g;
    }
    diagnostics_.proposed = true;

    // Scale residuals before dot products to avoid overflow/underflow. This is
    // one common scalar for all columns, so it does not change the LS problem.
    double residualScale = 0.0;
    for (const auto& h : history_)
        residualScale = std::max(residualScale, fileHistory() ? h.maximumResidual : h.f.cwiseAbs().maxCoeff());
    if (!(residualScale > 0.0) || !std::isfinite(residualScale)) {
        diagnostics_.reason = "invalid residual scale";
        retainLatest();
        return g;
    }
    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(columns, columns);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(columns);
    if (fileHistory()) fileGram(residualScale, gram, rhs);
    else {
    Eigen::VectorXd differenceI(stateSize_), differenceJ(stateSize_);
    for (int i = 0; i < columns; ++i) {
        differenceI = history_[i + 1].f / residualScale - history_[i].f / residualScale;
        // Avoid an N-by-m temporary; at most two N-vector differences are live.
        rhs[i] = differenceI.dot(history_.back().f / residualScale);
        for (int j = 0; j <= i; ++j) {
            if (j == i) gram(i, j) = differenceI.squaredNorm();
            else {
                differenceJ = history_[j + 1].f / residualScale - history_[j].f / residualScale;
                gram(i, j) = differenceI.dot(differenceJ);
            }
            gram(j, i) = gram(i, j);
        }
    }
    }
    if (!gram.allFinite() || !rhs.allFinite()) {
        diagnostics_.reason = "nonfinite history Gram matrix";
        retainLatest();
        return g;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectral(gram);
    if (spectral.info() != Eigen::Success) {
        diagnostics_.reason = "history spectral decomposition failed";
        retainLatest();
        return g;
    }
    const auto eigenvalues = spectral.eigenvalues();
    const double largest = eigenvalues[columns - 1];
    if (!(largest > std::numeric_limits<double>::min())) {
        diagnostics_.reason = "zero history residual differences";
        retainLatest();
        return g;
    }
    // The Gram spectrum squares the singular-value ratio. Drop unresolved
    // directions rather than amplifying their roundoff with an unregularized solve.
    const double rankThreshold = largest * 1e-12;
    const double ridge = relativeRegularization_ * largest;
    diagnostics_.regularization = ridge;
    Eigen::VectorXd spectralCoefficients = spectral.eigenvectors().transpose() * rhs;
    for (int i = 0; i < columns; ++i) {
        if (eigenvalues[i] > rankThreshold) {
            spectralCoefficients[i] /= eigenvalues[i] + ridge;
            ++diagnostics_.rank;
        } else spectralCoefficients[i] = 0.0;
    }
    const Eigen::VectorXd gamma = spectral.eigenvectors() * spectralCoefficients;
    diagnostics_.gammaNorm = gamma.stableNorm();
    if (diagnostics_.rank == 0 || !gamma.allFinite() || !std::isfinite(diagnostics_.gammaNorm)
        || diagnostics_.gammaNorm > gammaNormLimit_) {
        diagnostics_.reason = "rank or coefficient safeguard rejected proposal";
        retainLatest();
        return g;
    }

    // g_k - sum_i gamma_i*(g_{i+1}-g_i) is an affine combination of map outputs.
    // Thus any common affine linear constraint of the outputs is preserved up
    // to rounding. In CFD, a shared conservative flux must be included in state.
    Eigen::VectorXd affine = Eigen::VectorXd::Zero(columns + 1);
    affine[0] = gamma[0];
    for (int i = 1; i < columns; ++i) affine[i] = gamma[i] - gamma[i - 1];
    affine[columns] = 1.0 - gamma[columns - 1];
    diagnostics_.affineCoefficientNorm = affine.stableNorm();
    if (!std::isfinite(diagnostics_.affineCoefficientNorm)
        || diagnostics_.affineCoefficientNorm > 2.0 * gammaNormLimit_ + 1.0) {
        diagnostics_.reason = "affine coefficient safeguard rejected proposal";
        retainLatest();
        return g;
    }
    Eigen::VectorXd candidate = g;
    if (fileHistory()) fileCandidate(gamma, candidate);
    else for (int i = 0; i < columns; ++i)
        candidate.noalias() -= gamma[i] * (history_[i + 1].g - history_[i].g);
    if (!candidate.allFinite()) {
        diagnostics_.reason = "nonfinite accelerated candidate";
        retainLatest();
        return g;
    }
    diagnostics_.accepted = true;
    diagnostics_.reason = "algebraically admissible; caller must validate physical residuals";
    return candidate;
}

} // namespace simple
