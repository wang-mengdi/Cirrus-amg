#include "MetaData.h"

DriverMetaData::~DriverMetaData() {
	while (!output_threads.empty()) {
		auto join_ptr = output_threads.front();
		output_threads.pop();
		join_ptr->join();
	}
}

double DriverMetaData::Time_At_Frame(int frame) {
	return frame * time_per_frame;
}

void DriverMetaData::init(const fs::path& json_path, json& j) {
	std::string output_base_dir = Json::Value(j, "output_base_dir", std::string("output"));
	//base_path = fs::current_path() / fs::path(output_base_dir);
    base_path = fs::path(output_base_dir) / json_path.stem();

	fps = Json::Value(j, "fps", 25);
	cfl = Json::Value(j, "cfl", (double)1.0);
	time_per_frame = 1.0 / fps;
	min_step_frame_fraction = Json::Value(j, "min_step_frame_fraction", (double)0);

	first_frame = Json::Value(j, "first_frame", 0);

	//try to load snapshot
	if (first_frame != 0) {
		int last_snapshot = Last_Snapshot_Frame(first_frame);
		if (last_snapshot != 0) {
			Pass("Found snapshot at frame {}, load snapshot from {} and run from frame {}", last_snapshot, Snapshot_Path(last_snapshot).string(), last_snapshot + 1);
			first_frame = last_snapshot + 1;
			current_frame = last_snapshot;
		}
		else {
			Warn("No snapshots are found, automaitcally run from frame 0");
			first_frame = 0;
		}
	}

	last_frame = Json::Value(j, "last_frame", fps * 10);
	snapshot_stride = Json::Value(j, "snapshot_stride", 0);

	output_queue_size = Json::Value(j, "queue_size", 10);
}

void DriverMetaData::Append_Output_Thread(std::shared_ptr<std::thread> thread_ptr) {
	while (output_threads.size() >= output_queue_size) {
		auto join_ptr = output_threads.front();
		output_threads.pop();
		join_ptr->join();
	}
	output_threads.push(thread_ptr);
}

fs::path DriverMetaData::Current_VTS_Path(const std::string identifier)
{
	return base_path / fs::path(fmt::format("{}{:04d}.vts", identifier, current_frame));
}

fs::path DriverMetaData::Current_VTU_Path(const std::string identifier)
{
	return base_path / fs::path(fmt::format("{}{:04d}.vtu", identifier, current_frame));
}

fs::path DriverMetaData::Current_OBJ_Path(const std::string identifier)
{
	return base_path / fs::path(fmt::format("{}{:04d}.obj", identifier, current_frame));
}

bool DriverMetaData::Should_Snapshot(void)
{
	return current_frame != 0 && snapshot_stride != 0 && current_frame % snapshot_stride == 0;
}

fs::path DriverMetaData::Snapshot_Base_Path(void)
{
	return base_path / fs::path("snapshots");
}

fs::path DriverMetaData::Snapshot_Path(int frame)
{
	return Snapshot_Base_Path() / fs::path(fmt::format("{:04d}", frame));
}

fs::path DriverMetaData::Current_Snapshot_Path(void)
{
	return Snapshot_Path(current_frame);
}

int DriverMetaData::Last_Snapshot_Frame(int start_frame)
{
    fs::path snap_base = Snapshot_Base_Path();
    if (!fs::is_directory(snap_base)) return 0;

    std::vector<int> snapshots;

    for (const auto& entry : fs::directory_iterator(snap_base)) {

        std::string name;

        if (fs::is_directory(entry.status())) {
            // old style: folder snapshot
            name = entry.path().filename().string();
        }
        else if (entry.path().extension() == ".bin") {
            // new style: %04d.bin
            name = entry.path().stem().string();
        }
        else {
            continue;
        }

        try {
            int frame = std::stoi(name);
            snapshots.push_back(frame);
        }
        catch (...) {
            // ignore non-numeric names
        }
    }

    if (snapshots.empty()) return 0;

    std::sort(snapshots.begin(), snapshots.end());

    auto it = std::lower_bound(snapshots.begin(), snapshots.end(), start_frame);

    if (it != snapshots.begin()) {
        --it;
        return *it;
    }

    return 0;
}
