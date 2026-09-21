// pi-lens-ignore: fatal, clang:pp_file_not_found
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cassert>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <SDL.h>
#include <cstdio>

#include "SexyAppFramework/graphics/GLImage.h"
#include "SexyAppFramework/graphics/GLInterface.h"
#include "GameConstants.h"
#include "LawnApp.h"
#include "Resources.h"
#include "Sexy.TodLib/TodStringFile.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/Challenge.h"
#include "Lawn/Board.h"
#include "Lawn/Coin.h"
#include "Lawn/System/ProfileMgr.h"
#include "SexyAppFramework/widget/WidgetManager.h"
#include "Lawn/Widget/SeedChooserScreen.h"
#include "Lawn/Widget/TitleScreen.h"
#include "Lawn/Zombie.h"
#include "Lawn/Plant.h"
#include "Lawn/LawnMower.h"
#include "ConstEnums.h"

namespace py = pybind11;
using namespace Sexy;

bool (*gAppCloseRequest)();
bool (*gAppHasUsedCheatKeys)();
std::string (*gGetCurrentLevelName)();

// Global cooldown scale for RL training. 1.0 = real PvZ cooldowns
// (sunflower 7.5s, melon 30s). 0.13 = 100cs (~1 step) for cheap plants.
// Modify via set_cooldown_scale(float). Default 1.0 (real cooldowns).
static float gCooldownScale = 1.0f;

// ── Action layout for Discrete(496) ──────────────────────────────
//   [0]        : Wait (do nothing this step)
//   [1..45]    : Shovel at grid (row, col)  → index = 1 + row*9 + col
//   [46..495]  : Plant seed i at (row, col) → index = 46 + i*45 + row*9 + col
//                                                                  i ∈ [0,9], row ∈ [0,4], col ∈ [0,8]
static constexpr int ACT_WAIT   = 0;
static constexpr int SHOVEL_OFF = 1;   // 45 entries
static constexpr int PLANT_OFF  = 46;  // 450 entries
static constexpr int NUM_ACTIONS = 496;
static constexpr int GRID_ROWS = 5;    // day board — no pool
static constexpr int GRID_COLS = 9;
static constexpr int NUM_SEEDS = 10;
static constexpr int FRAMES_PER_STEP = 100;
static constexpr int MAX_STEPS = 2000;
static constexpr int NZOMBIE_TYPES = 33;
static constexpr int PLANT_CHANNELS = 3;
static constexpr int SPATIAL_CHANNELS = PLANT_CHANNELS + NZOMBIE_TYPES;
static constexpr int WAVE_PER_STAGE = 20;
static constexpr int MAX_START_WAVE = 100;

// Curriculum starting sun: harder waves get more resources to maintain.
// Phase 0 (wave 1) gets 1000 sun — enough for a real early build order
// (e.g. 2 sunflowers + 2 peashooters across lanes) so the agent can
// experiment with multi-lane defense rather than starving after one peashooter.
static int startingSun(int wave) {
    if (wave >= 50) return 9990;
    if (wave >= 30) return 6000;
    if (wave >= 20) return 3000;
    if (wave >= 10) return 1500;
    return 1000;
}

class PvZEnv {
public:
    int mStepCount = 0;
    int mPrevTriggeredMowers = 0;
    int mPrevPlantCount = 0;
    int mLastAction = 0;
    float mPlantBonusScale = 0.3f;
    float mSunPenaltyScale = 0.0f;
    int mPrevSunMoney = 0;
    int mPrevWaves = 0;
    float mPrevThreatPotential = 0.0f;
    float mRewardDamage = 0.0f;
    float mRewardWave = 0.0f;
    float mRewardAlive = 0.0f;
    float mRewardPlant = 0.0f;
    float mRewardShovel = 0.0f;
    float mRewardSun = 0.0f;
    float mRewardDanger = 0.0f;
    float mRewardMower = 0.0f;
    float mRewardDeath = 0.0f;
    float mRewardTime = 0.0f;
    bool mChainStages = false;
    // Observation version. v1 (default): 36 spatial channels / 24 globals —
    // every existing checkpoint. v2: +2 spatial channels (36-37 = the
    // normal-plant layer from GetPlantsOnLawn, revealing the plant hidden
    // under a Pumpkin shell, immune to the mPlants iteration-order
    // overwrite that decides which plant owns channels 0-2 on a stacked
    // cell) and +2 globals (24 = absolute wave/(wavesPerStage+1),
    // 25 = survival stage — under chain_stages the per-stage wave fraction
    // resets at every boundary and the stage index is otherwise
    // unobservable even though zombie strength scales with it).
    int mObsVersion = 1;
    // Deck-as-config: set_deck() overrides the default meta-deck; applies
    // on the next reset() (every packet/cost/mask path already iterates
    // the seed bank generically, so no other change is needed).
    SeedType mDeck[NUM_SEEDS] = {
        SeedType::SEED_SUNFLOWER,    SeedType::SEED_TWINSUNFLOWER,
        SeedType::SEED_MELONPULT,    SeedType::SEED_WINTERMELON,
        SeedType::SEED_GLOOMSHROOM,  SeedType::SEED_FUMESHROOM,
        SeedType::SEED_PUMPKINSHELL, SeedType::SEED_GARLIC,
        SeedType::SEED_SQUASH,       SeedType::SEED_JALAPENO
    };

    PvZEnv(const std::string& resdir = "pvz-portable/", const std::string& savedir = "pvz-portable/savedata/") {
        putenv((char*)"SDL_AUDIODRIVER=dummy");

        TodStringListSetColors(gLawnStringFormats, gLawnStringFormatCount);
        gGetCurrentLevelName = LawnGetCurrentLevelName;
        gAppCloseRequest = LawnGetCloseRequest;
        gAppHasUsedCheatKeys = LawnHasUsedCheatKeys;
        gExtractResourcesByName = Sexy::ExtractResourcesByName;

        // One engine per process: constructing a second PvZEnv must reuse
        // the existing gLawnApp instead of re-running LawnApp::Init() over
        // live global engine state (the resource reload segfaults).
        // resdir/savedir of later constructions are ignored.
        if (gLawnApp) return;

        gLawnApp = new LawnApp();

        std::string resArg = "-resdir=" + resdir;
        std::string saveArg = "-savedir=" + savedir;
        mResArgBuf.assign(resArg.begin(), resArg.end());
        mResArgBuf.push_back('\0');
        mSaveArgBuf.assign(saveArg.begin(), saveArg.end());
        mSaveArgBuf.push_back('\0');
        char* argv[] = { (char*)"pvz", mResArgBuf.data(), mSaveArgBuf.data(), NULL };
        gLawnApp->SetArgs(3, argv);

        gLawnApp->Init();

        gLawnApp->mRunning = true;
        ProfileMgr* profileMgr = gLawnApp->mProfileMgr;
        if (!gLawnApp->mPlayerInfo)
            gLawnApp->mPlayerInfo = profileMgr->AddProfile("RLAgent");
        gLawnApp->LoadingThreadProc();
        gLawnApp->mLoadingThreadCompleted = true;
        gLawnApp->mLoaded = true;

        // The normal app removes the loading/title widget in LoadingCompleted().
        // The synchronous headless bootstrap bypasses that callback, leaving the
        // title screen above the board and making framebuffer captures look stuck
        // on the PopCap logo even while gameplay updates normally.
        if (gLawnApp->mTitleScreen) {
            WidgetManager* widgetManager = gLawnApp->mWidgetManager;
            widgetManager->RemoveWidget(gLawnApp->mTitleScreen);
            delete gLawnApp->mTitleScreen;
            gLawnApp->mTitleScreen = nullptr;
        }
    }

private:
    std::vector<char> mResArgBuf;
    std::vector<char> mSaveArgBuf;

public:

    ~PvZEnv() {
        if (gLawnApp) {
            gLawnApp->Shutdown();
            delete gLawnApp;
            gLawnApp = nullptr;
        }
    }

    void set_plant_bonus_scale(float scale) {
        mPlantBonusScale = scale;
    }

    void reset(int wave, int seed = -1) {
        mStepCount = 0;
        mPrevTriggeredMowers = 0;
        mPrevThreatPotential = 0.0f;
        mPrevSunMoney = 0;
        mPrevWaves = 0;
        int survivalStage = std::max(0, (wave - 1) / WAVE_PER_STAGE);
        int waveIndex = (wave - 1) % WAVE_PER_STAGE;

        gLawnApp->mGameMode = GameMode::GAMEMODE_SURVIVAL_ENDLESS_STAGE_1;
        gLawnApp->KillBoard();
        // KillBoard defers the old Board widget to the app's safe-delete
        // list, which the normal main loop (SexyAppBase::Process) drains;
        // this bridge never runs that loop, so drain it here. reset() runs
        // outside the widget update loop, so deletion is safe here.
        gLawnApp->DrainSafeDeleteList();
        if (seed >= 0) {
            gLawnApp->mAppRandSeed = seed;
            gLawnApp->mPlayerInfo->mId = seed;
            Sexy::SRand(static_cast<unsigned long>(seed));
        }
        gLawnApp->mBoard = new Board(gLawnApp);
        gLawnApp->mBoard->Resize(0, 0, gLawnApp->mWidth, gLawnApp->mHeight);
        gLawnApp->mWidgetManager->AddWidget(gLawnApp->mBoard);
        gLawnApp->mWidgetManager->BringToBack(gLawnApp->mBoard);
        gLawnApp->mWidgetManager->SetFocus(gLawnApp->mBoard);

        // Set survival stage and deterministic level seed before InitLevel(),
        // because survival wave tables are generated during initialization.
        gLawnApp->mBoard->mChallenge->mSurvivalStage = survivalStage;
        if (seed >= 0) {
            gLawnApp->mBoard->mBoardRandSeed = seed;
            Sexy::SRand(static_cast<unsigned long>(seed));
            std::srand(static_cast<unsigned int>(seed));
        }

        gLawnApp->mBoard->InitLevel();
        if (seed >= 0) {
            // Reset every RNG stream after engine initialization. InitLevel uses
            // the global stream for wave generation; gameplay then starts from
            // the same state in every seeded reset.
            gLawnApp->mBoard->mBoardRandSeed = seed;
            gLawnApp->mBoard->mGameID = seed;
            Sexy::SRand(static_cast<unsigned long>(seed));
            std::srand(static_cast<unsigned int>(seed));
        }

        // Seed bank slots come from mDeck: the default meta-deck unless
        // set_deck() overrode it (applied here so packets, costs and the
        // action mask — which already iterate the seed bank generically —
        // all pick the custom deck up together).
        SeedBank* bank = gLawnApp->mBoard->mSeedBank;
        bank->mNumPackets = NUM_SEEDS;
        for (int i = 0; i < NUM_SEEDS; i++) {
            bank->mSeedPackets[i].SetPacketType(mDeck[i]);
        }
        // SetPacketType marks plants with refresh=5000 (melon, winter, gloom)
        // as mActive=false, mRefreshing=true. That puts them on a 35-80 RL
        // step cooldown from t=0. RefreshAllPackets force-activates them so
        // the agent can use any plant at the start. After the first planting
        // the normal WasPlanted cooldown kicks in.
        bank->RefreshAllPackets();
        // Scale initial cooldowns for RL training. SetPacketType() set the
        // heavy 2000/8000 cs values for plants like winter-melon; we override
        // them here. RefreshAllPackets above activates them so the agent can
        // use any plant at t=0 with the scaled cooldown.
        if (gCooldownScale != 1.0f) {
            for (int i = 0; i < bank->mNumPackets; i++) {
                SeedPacket& p = bank->mSeedPackets[i];
                int newRefresh = (int)(p.mRefreshTime * gCooldownScale);
                if (newRefresh < 100 && p.mRefreshTime > 0) newRefresh = 100;
                p.mRefreshTime = newRefresh;
                p.mRefreshCounter = 0;
            }
        }
        // Apply curriculum starting resources and wave position.
        gLawnApp->mBoard->mSunMoney = startingSun(wave);
        gLawnApp->mBoard->mCurrentWave = waveIndex;
        gLawnApp->mBoard->mTotalSpawnedWaves = wave - 1;
        gLawnApp->mBoard->mZombieCountDown = ZOMBIE_COUNTDOWN_FIRST_WAVE;
        gLawnApp->mBoard->mZombieCountDownStart = ZOMBIE_COUNTDOWN_FIRST_WAVE;
        if (seed >= 0) {
            // Keep the first sky-sun schedule independent of any cosmetic RNG
            // consumed after InitLevel (deck setup, particles, animations).
            gLawnApp->mBoard->mSunCountDown = 425 + seed % 276;
        }
        mPrevSunMoney = gLawnApp->mBoard->mSunMoney;
        mPrevWaves = wave;

        // The normal game creates lawnmowers during the level-intro cutscene
        // (CutScene::PlaceLawnItems -> Board::InitLawnMowers). This bridge
        // skips the cutscene, so create them explicitly on the fresh board
        // and roll them into place exactly like the engine does: the
        // MOWER_ROLLING_IN state animates x from -160 to -21 over 100 frames
        // (one RL step at FRAMES_PER_STEP=100). Without this, every episode
        // ran with zero mowers: any single lane leak was an instant loss.
        gLawnApp->mBoard->InitLawnMowers();
        LawnMower* aMower = nullptr;
        while (gLawnApp->mBoard->IterateLawnMowers(aMower)) {
            aMower->mMowerState = LawnMowerState::MOWER_ROLLING_IN;
            aMower->mRollingInCounter = 0;
            aMower->mVisible = true;
        }

        gLawnApp->mGameScene = GameScenes::SCENE_PLAYING;
    }

    // ── Auto-collect all sun coins on the board ──
    void autoCollectSun() {
        Board* board = gLawnApp->mBoard;
        if (!board) return;

        Coin* coin = nullptr;
        while (board->mCoins.IterateNext(coin)) {
            if (!coin->mDead && coin->IsSun() && !coin->mIsBeingCollected) {
                coin->ScoreCoin();
            }
        }
    }

    // ── Execute an action via the game's internal API ──
    void executeAction(int action) {
        if (action == ACT_WAIT) return;

        Board* board = gLawnApp->mBoard;
        if (!board) return;

        if (action >= SHOVEL_OFF && action < SHOVEL_OFF + 45) {
            int idx = action - SHOVEL_OFF;
            int row = idx / GRID_COLS;
            int col = idx % GRID_COLS;
            Plant* p = board->GetTopPlantAt(col, row, PlantPriority::TOPPLANT_ANY);
            if (p) {
                p->Die();
                board->mPlantsShoveled++;
            }
            return;
        }

        // Cob cannon actions removed; agent learns to plant from scratch.
        if (action >= PLANT_OFF && action < PLANT_OFF + 450) {
            int idx = action - PLANT_OFF;
            int seedIdx = idx / (GRID_ROWS * GRID_COLS);
            int row = (idx % (GRID_ROWS * GRID_COLS)) / GRID_COLS;
            int col = (idx % (GRID_ROWS * GRID_COLS)) % GRID_COLS;

            if (seedIdx >= board->mSeedBank->mNumPackets) return;

            SeedPacket& packet = board->mSeedBank->mSeedPackets[seedIdx];
            SeedType seedType = packet.mPacketType;

            // Validate
            if (!packet.CanPickUp()) return;
            if (board->CanPlantAt(col, row, seedType) != PlantingReason::PLANTING_OK) return;

            int cost = board->GetCurrentPlantCost(seedType, SeedType::SEED_NONE);
            if (!board->CanTakeSunMoney(cost)) return;

            // Execute plant
            board->TakeSunMoney(cost);
            packet.Deactivate();

            // Upgrades must consume their base plant, exactly like the UI
            // path (Board::MouseDownWithSeedPacket): Twin Sunflower eats
            // Sunflower, Winter Melon eats Melon-pult, Gloom-shroom eats
            // Fume-shroom (carrying the awake state over), and a Pumpkin
            // replant eats the old Pumpkin. Calling AddPlant alone leaves
            // the base plant alive under the upgrade, inflating the economy
            // and double-counting the cell in observations.
            bool aIsAwake = false;
            int aWakeUpCounter = 0;
            PlantsOnLawn aPlantsOnLawn;
            board->GetPlantsOnLawn(col, row, &aPlantsOnLawn);
            Plant* aNormalPlant = aPlantsOnLawn.mNormalPlant;
            Plant* aPumpkinPlant = aPlantsOnLawn.mPumpkinPlant;
            if (aNormalPlant && aNormalPlant->IsUpgradableTo(seedType)) {
                if (seedType == SeedType::SEED_GLOOMSHROOM) {
                    aIsAwake = !aNormalPlant->mIsAsleep;
                    aWakeUpCounter = aNormalPlant->mWakeUpCounter;
                }
                aNormalPlant->Die();
            }
            if (seedType == SeedType::SEED_PUMPKINSHELL && aPumpkinPlant &&
                aPumpkinPlant->mSeedType == SeedType::SEED_PUMPKINSHELL) {
                aPumpkinPlant->Die();
            }

            Plant* aNewPlant = board->AddPlant(col, row, seedType, SeedType::SEED_NONE);
            if (aNewPlant) {
                if (aIsAwake) {
                    aNewPlant->SetSleeping(false);
                } else {
                    aNewPlant->mWakeUpCounter = aWakeUpCounter;
                }
            }
            packet.WasPlanted();
            // Apply RL training cooldown scale. WasPlanted sets mRefreshTime
            // to the real game value; we override it here so cooldowns can
            // be tightened for the agent. gCooldownScale=1.0 means real PvZ.
            if (gCooldownScale != 1.0f) {
                int newRefresh = (int)(packet.mRefreshTime * gCooldownScale);
                if (newRefresh < 100) newRefresh = 100;  // floor at 1 step
                packet.mRefreshTime = newRefresh;
                packet.mRefreshCounter = 0;  // restart cooldown
            }
        }
    }

    // ── Render the current board into an RGB NumPy array ──
    py::array_t<unsigned char> captureFrame() {
        // Query the actual drawable size of the hidden window.
        int w = 0, h = 0;
        if (gLawnApp && gLawnApp->mWindow) {
            SDL_GL_GetDrawableSize((SDL_Window*)gLawnApp->mWindow, &w, &h);
        }
        if (w <= 0 || h <= 0) {
            w = BOARD_WIDTH;
            h = BOARD_HEIGHT;
        }

        // Update viewport and presentation rect, then tell the widget manager
        // about the current coordinate system. This mirrors the normal setup
        // in Window.cpp / SexyAppBase::InitGLInterface.
        GLInterface* glInterface = gLawnApp->mGLInterface;
        WidgetManager* widgetManager = gLawnApp->mWidgetManager;
        glInterface->UpdateViewport();
        widgetManager->Resize(gLawnApp->mScreenBounds, glInterface->mPresentationRect);

        // Ensure we are writing to the default framebuffer.
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw the current widget hierarchy to the backbuffer.
        widgetManager->mImage = glInterface->GetScreenImage();
        widgetManager->MarkAllDirty();
        widgetManager->DrawScreen();

        glFlush();

        // Read back RGBA pixels.
        std::vector<unsigned char> rgba(w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

        // Convert RGBA bottom-up to RGB top-down.
        py::array_t<unsigned char> frame({h, w, 3});
        unsigned char* dst = frame.mutable_data();
        for (int y = 0; y < h; ++y) {
            const unsigned char* srcRow = &rgba[(h - 1 - y) * w * 4];
            unsigned char* dstRow = &dst[y * w * 3];
            for (int x = 0; x < w; ++x) {
                dstRow[x * 3 + 0] = srcRow[x * 4 + 0];
                dstRow[x * 3 + 1] = srcRow[x * 4 + 1];
                dstRow[x * 3 + 2] = srcRow[x * 4 + 2];
            }
        }
        return frame;
    }


    py::list squashTargetableLanes() {
        py::list result;
        Board* board = gLawnApp ? gLawnApp->mBoard : nullptr;
        for (int row = 0; row < 5; ++row) {
            result.append(board && Plant::FindSquashTargetAt(
                board, gLawnApp, row, board->GridToPixelX(6, row), 80
            ) != nullptr);
        }
        return result;
    }

    py::list squashTelemetry() {
        py::list telemetry;
        Board* board = gLawnApp ? gLawnApp->mBoard : nullptr;
        if (!board) return telemetry;

        Plant* plant = nullptr;
        while (board->mPlants.IterateNext(plant)) {
            if (plant->mSeedType != SeedType::SEED_SQUASH) continue;
            py::dict item;
            item["row"] = plant->mRow;
            item["column"] = plant->mPlantCol;
            item["state"] = static_cast<int>(plant->mState);
            item["state_countdown"] = plant->mStateCountdown;
            item["target_id"] = static_cast<unsigned int>(plant->mTargetZombieID);
            item["target_x"] = plant->mTargetX;
            item["plant_x"] = plant->mX;
            item["plant_y"] = plant->mY;
            item["damage"] = plant->mRLSquashDamage;
            item["hits"] = plant->mRLSquashHits;

            Zombie* target = board->ZombieTryToGet(plant->mTargetZombieID);
            if (target) {
                item["target_row"] = target->mRow;
                item["target_x_live"] = target->mPosX;
                item["target_hp"] = target->mBodyHealth + target->mHelmHealth + target->mShieldHealth;
                item["target_phase"] = static_cast<int>(target->mZombiePhase);
            } else {
                item["target_row"] = py::none();
                item["target_x_live"] = py::none();
                item["target_hp"] = py::none();
                item["target_phase"] = py::none();
            }
            telemetry.append(item);
        }
        return telemetry;
    }

    // ── Endless chaining: advance to the next survival stage in place ──
    // Preempts the engine's stock stage transition (the UI path destroys
    // and reinitializes widgets, unsafe under this headless bridge) by
    // resetting the wave state machine on the live board: plants, sun and
    // the seed bank persist across the stage boundary. Stage must be bumped
    // BEFORE InitZombieWaves() because wave tables are generated from
    // mSurvivalStage during initialization; difficulty then scales as
    // zombiePoints = (stage * wavesPerStage + wave) * 2 / 5 + 1.
    void advanceSurvivalStage() {
        Board* board = gLawnApp->mBoard;
        board->mNextSurvivalStageCounter = 0;
        board->mLevelComplete = false;
        if (gLawnApp->mGameMode < GameMode::GAMEMODE_SURVIVAL_ENDLESS_STAGE_5)
            gLawnApp->mGameMode = static_cast<GameMode>(gLawnApp->mGameMode + 1);
        board->mChallenge->mSurvivalStage++;
        board->InitZombieWaves();
        board->mSeedBank->RefreshAllPackets();
    }

    void setChainStages(bool chain) { mChainStages = chain; }

    void setObsVersion(int v) {
        if (v != 1 && v != 2)
            throw std::invalid_argument("obs_version must be 1 or 2");
        mObsVersion = v;
    }

    // Override the 10-slot deck (seed-type integers; range [0, 44]).
    // Applies on the next reset().
    void setDeck(std::vector<int> seeds) {
        if ((int)seeds.size() != NUM_SEEDS)
            throw std::invalid_argument("deck must contain exactly 10 seed types");
        for (int s : seeds) {
            if (s < 0 || s > 44)
                throw std::invalid_argument("seed type out of range [0, 44]");
        }
        for (int i = 0; i < NUM_SEEDS; i++)
            mDeck[i] = static_cast<SeedType>(seeds[i]);
    }

    // Test hook: inject the stage-end countdown (mid-countdown value, as
    // the engine leaves it right after LevelWon) so the chaining path can be
    // exercised deterministically without playing a full winning stage.
    // Values far from zero matter: the countdown only reaches TryToSaveGame
    // at ==1 and mLevelComplete at ==0, and chaining preempts far above that.
    void debugTriggerStageEnd() {
        gLawnApp->mBoard->mNextSurvivalStageCounter = 400;
    }

    // ── Shared state update after the action has been executed ──
    py::tuple buildStepResult() {
        Board* board = gLawnApp->mBoard;

        float reward = computeReward();
        bool lost = (gLawnApp->mGameScene == GameScenes::SCENE_ZOMBIES_WON);
        // Endless Survival begins a 500-frame stage-transition sequence after
        // wave 20. The stock UI path destroys/reinitializes board widgets,
        // which is unsafe while this headless bridge continues stepping the
        // same Python-owned board. Without chaining, end the RL episode
        // before that transition can execute; the next reset creates a
        // fresh board deliberately.
        bool stageComplete = board->mLevelComplete || board->mNextSurvivalStageCounter > 0;
        // Endless chaining: preempt the unsafe transition by advancing the
        // live board to the next survival stage in place. stage_complete is
        // still reported for this step (per-stage bonus); the episode
        // continues with persisted plants, sun and seed bank.
        if (stageComplete && !lost && mChainStages)
            advanceSurvivalStage();
        bool done = lost || (stageComplete && !mChainStages);
        bool truncated = (mStepCount >= MAX_STEPS);

        py::dict obs = get_obs();
        py::array_t<bool> mask = get_action_mask();
        py::dict info;
        int wavesPerStage = board->GetNumWavesPerSurvivalStage();
        int absoluteWave = board->mChallenge->mSurvivalStage * wavesPerStage + board->mCurrentWave + 1;

        info["sun"] = board->mSunMoney;
        info["wave"] = absoluteWave;
        info["stage"] = board->mChallenge->mSurvivalStage;
        info["step"] = mStepCount;
        info["triggered_mowers"] = board->mTriggeredLawnMowers;
        info["zombie_min_x"] = minZombieX();
        info["num_waves"] = board->mNumWaves;
        info["stage_complete"] = stageComplete;
        info["lost"] = lost;
        info["reward_damage"] = mRewardDamage;
        info["reward_wave"] = mRewardWave;
        info["reward_alive"] = mRewardAlive;
        info["reward_plant"] = mRewardPlant;
        info["reward_shovel"] = mRewardShovel;
        info["reward_sun"] = mRewardSun;
        info["reward_danger"] = mRewardDanger;
        info["reward_mower"] = mRewardMower;
        info["reward_death"] = mRewardDeath;
        info["reward_time"] = mRewardTime;
        info["rand_state"] = py::bytes(Sexy::GetRandState());
        info["squash_targetable_lanes"] = squashTargetableLanes();
        info["squash_telemetry"] = squashTelemetry();

        return py::make_tuple(obs, mask, reward, done, truncated, info);
    }

    // ── Step: execute action, advance frames, return full tuple ──
    py::tuple step(int action) {
        Board* board = gLawnApp->mBoard;

        // Snapshot pre-step state for reward
        mPrevTriggeredMowers = board->mTriggeredLawnMowers;
        mPrevThreatPotential = threatPotential();
        mPrevWaves = board->mCurrentWave;
        mPrevPlantCount = plantCount();
        mLastAction = action;

        // Execute the action
        executeAction(action);

        // Advance the game FRAMES_PER_STEP frames.
        for (int i = 0; i < FRAMES_PER_STEP; i++) {
            gLawnApp->DoUpdateFrames();
            autoCollectSun();
        }

        mStepCount++;
        return buildStepResult();
    }

    // ── Step with recording: also return frames captured during the step ──
    py::tuple record_step(int action, int frameInterval = 5) {
        if (frameInterval <= 0) frameInterval = 1;
        if (frameInterval > FRAMES_PER_STEP) frameInterval = FRAMES_PER_STEP;

        Board* board = gLawnApp->mBoard;

        // Snapshot pre-step state for reward
        mPrevTriggeredMowers = board->mTriggeredLawnMowers;
        mPrevThreatPotential = threatPotential();
        mPrevWaves = board->mCurrentWave;
        mPrevPlantCount = plantCount();
        mLastAction = action;

        // Execute the action
        executeAction(action);

        py::list frames;
        py::list telemetryFrames;

        // Advance FRAMES_PER_STEP frames, capturing frames at the requested interval.
        for (int i = 0; i < FRAMES_PER_STEP; i++) {
            gLawnApp->DoUpdateFrames();
            autoCollectSun();

            if ((i + 1) % frameInterval == 0) {
                frames.append(captureFrame());
                telemetryFrames.append(squashTelemetry());
            }
        }

        mStepCount++;
        py::tuple result = buildStepResult();
        py::dict resultInfo = result[5].cast<py::dict>();
        resultInfo["squash_telemetry_frames"] = telemetryFrames;
        return py::make_tuple(result[0], result[1], result[2], result[3], result[4], resultInfo, frames);
    }

    // ── Step while sampling Squash internals without rendering frames ──
    py::tuple telemetry_step(int action, int frameInterval = 5) {
        if (frameInterval <= 0) frameInterval = 1;
        if (frameInterval > FRAMES_PER_STEP) frameInterval = FRAMES_PER_STEP;

        Board* board = gLawnApp->mBoard;
        mPrevTriggeredMowers = board->mTriggeredLawnMowers;
        mPrevThreatPotential = threatPotential();
        mPrevWaves = board->mCurrentWave;
        mPrevPlantCount = plantCount();
        mLastAction = action;
        executeAction(action);

        py::list telemetryFrames;
        for (int i = 0; i < FRAMES_PER_STEP; i++) {
            gLawnApp->DoUpdateFrames();
            autoCollectSun();
            if ((i + 1) % frameInterval == 0) {
                telemetryFrames.append(squashTelemetry());
            }
        }

        mStepCount++;
        py::tuple result = buildStepResult();
        py::dict resultInfo = result[5].cast<py::dict>();
        resultInfo["squash_telemetry_frames"] = telemetryFrames;
        return py::make_tuple(result[0], result[1], result[2], result[3], result[4], resultInfo);
    }

    float computeReward() {
        Board* board = gLawnApp->mBoard;

        float reward = 0.0f;
        mRewardDamage = 0.0f;
        mRewardWave = 0.0f;
        mRewardAlive = 0.0f;
        mRewardPlant = 0.0f;
        mRewardShovel = 0.0f;
        mRewardSun = 0.0f;
        mRewardDanger = 0.0f;
        mRewardMower = 0.0f;
        mRewardDeath = 0.0f;
        mRewardTime = 0.0f;

        // 1. Threat-potential shaping. Raw total-HP deltas charged zombie
        // spawns to whichever action happened to precede the 100-frame burst.
        // This potential credits state transitions near the house instead.
        float threatDelta = mPrevThreatPotential - 0.999f * threatPotential();
        if (gLawnApp->mGameScene == GameScenes::SCENE_ZOMBIES_WON) {
            // Terminal potential is zero: a lost episode receives no accidental
            // positive shaping for the board state that caused the loss.
            threatDelta = mPrevThreatPotential;
        }
        mRewardDamage = std::clamp(0.01f * threatDelta, -1.0f, 1.0f);
        reward += mRewardDamage;

        // 2. Wave progression (survival). Dominant sparse signal for advancing.
        // Strongly reward reaching higher waves; this is the main objective.
        int waveDelta = board->mCurrentWave - mPrevWaves;
        if (waveDelta > 0) {
            mRewardWave = waveDelta * 10.0f;
            reward += mRewardWave;
        }
        int wavesPerStage = board->GetNumWavesPerSurvivalStage();
        int absoluteWave = board->mChallenge->mSurvivalStage * wavesPerStage + board->mCurrentWave + 1;
        // Per-step alive bonus: 0.05 per wave reached. Dense signal that
        // grows with progress; forces the agent to defend (not plant and wait).
        // (Old uniform 0.3*delta bonus removed; replaced by seed-weighted
        // bonus below to push the agent toward attack plants.)
        mRewardAlive = 0.05f * (float)(absoluteWave);
        reward += mRewardAlive;
        int plantDelta = plantCount() - mPrevPlantCount;
        // Shovel penalty: prevent the shovel-then-replant reward hack.
        // Without this, the agent plants a cheap seed, immediately shovels
        // it, and replants in the same cell to farm +0.3 per step.
        if (mLastAction >= SHOVEL_OFF && mLastAction < SHOVEL_OFF + 45) {
            mRewardShovel = -0.5f;
            reward += mRewardShovel;
        }
        // Seed-type-weighted per-plant bonus. Attack plants (melon=2, winter=3,
        // squash=8, jalapeno=9) get 2-4x more reward than economy (sun=0, twin=1).
        // This forces the agent to discover that killing zombies > hoarding sun.
        if (mLastAction >= PLANT_OFF && plantDelta > 0) {
            int seedIdx = (mLastAction - PLANT_OFF) / (GRID_ROWS * GRID_COLS);
            float seedWeight = 1.0f;
            if (seedIdx == 2 || seedIdx == 3) seedWeight = 4.0f;      // melon, winter melon
            else if (seedIdx == 8 || seedIdx == 9) seedWeight = 3.0f; // squash, jalapeno
            else if (seedIdx == 6 || seedIdx == 7) seedWeight = 2.0f; // pumpkin, garlic
            mRewardPlant = mPlantBonusScale * seedWeight * plantDelta;
            reward += mRewardPlant;
        }

        // Configured as an opt-in term; the control policy keeps this at zero
        // because penalizing saved sun can suppress the economy build order.
        mRewardSun = -mSunPenaltyScale * (float)board->mSunMoney;
        reward += mRewardSun;

        // 3. Danger-zone penalty: sustained pressure while any living zombie is
        // within 2 tiles of the house (mPosX < 200px). This is a direct penalty
        // for letting zombies get close, independent of the distance reward.
        int dangerCount = zombiesInDangerZone();
        mRewardDanger = -0.05f * dangerCount;
        reward += mRewardDanger;

        // 4. Lawnmower penalty.
        int mowerDelta = board->mTriggeredLawnMowers - mPrevTriggeredMowers;
        if (mowerDelta > 0) {
            mRewardMower = -0.1f * mowerDelta;
            reward += mRewardMower;
        }

        // 5. Brains eaten.
        if (gLawnApp->mGameScene == GameScenes::SCENE_ZOMBIES_WON) {
            mRewardDeath = -10.0f;
            reward += mRewardDeath;
        }

        // 6. Uniform time penalty.
        mRewardTime = -0.02f;
        reward += mRewardTime;

        // Threat shaping is clipped; the remaining terms retain their explicit
        // scales and Python may normalize the combined reward for PPO.
        return reward;
    }

    float totalZombieHP() {
        Board* board = gLawnApp->mBoard;
        if (!board) return 0.0f;
        float total = 0.0f;
        Zombie* z = nullptr;
        while (board->mZombies.IterateNext(z)) {
            total += (float)(z->mBodyHealth + z->mHelmHealth + z->mShieldHealth);
        }
        return total;
    }

    float threatPotential() {
        Board* board = gLawnApp->mBoard;
        if (!board) return 0.0f;
        float potential = 0.0f;
        Zombie* z = nullptr;
        while (board->mZombies.IterateNext(z)) {
            float hp = (float)(z->mBodyHealth + z->mHelmHealth + z->mShieldHealth);
            if (hp <= 0.0f) continue;
            hp = std::min(hp, 1000.0f);
            potential += hp * std::exp(-std::max(0.0f, z->mPosX) / 400.0f);
        }
        return potential;
    }

    int plantCount() {
        Board* board = gLawnApp->mBoard;
        if (!board) return 0;
        int count = 0;
        Plant* p = nullptr;
        while (board->mPlants.IterateNext(p)) {
            if (!p->mDead) {
                count++;
            }
        }
        return count;
    }
    // Count living zombies within 2 tiles of the house (X < 200px).
    int zombiesInDangerZone() {
        Board* board = gLawnApp->mBoard;
        if (!board) return 0;
        int count = 0;
        Zombie* z = nullptr;
        while (board->mZombies.IterateNext(z)) {
            float hp = (float)(z->mBodyHealth + z->mHelmHealth + z->mShieldHealth);
            if (hp > 0.0f && z->mPosX < 200.0f) {
                count++;
            }
        }
        return count;
    }

    float minZombieX() {
        Board* board = gLawnApp->mBoard;
        if (!board) return 0.0f;
        float minX = 100000.0f;
        bool found = false;
        Zombie* z = nullptr;
        while (board->mZombies.IterateNext(z)) {
            float hp = (float)(z->mBodyHealth + z->mHelmHealth + z->mShieldHealth);
            if (hp > 0.0f) {
                if (!found || z->mPosX < minX) {
                    minX = z->mPosX;
                    found = true;
                }
            }
        }
        return found ? minX : 0.0f;
    }

    py::array_t<bool> get_action_mask() {
        py::array_t<bool> mask({NUM_ACTIONS});
        auto buf = mask.mutable_unchecked<1>();
        std::memset(mask.mutable_data(), 0, NUM_ACTIONS * sizeof(bool));

        Board* board = gLawnApp->mBoard;
        if (!board) return mask;

        // Wait is always valid
        buf(ACT_WAIT) = true;

        // Shovel: valid if there's a plant at the grid cell
        for (int row = 0; row < GRID_ROWS; row++) {
            for (int col = 0; col < GRID_COLS; col++) {
                Plant* p = board->GetTopPlantAt(col, row, PlantPriority::TOPPLANT_ANY);
                if (p) {
                    buf(SHOVEL_OFF + row * GRID_COLS + col) = true;
                }
            }
        }
        // Plant: valid if seed pickable AND can plant at grid
        for (int s = 0; s < NUM_SEEDS && s < board->mSeedBank->mNumPackets; s++) {
            SeedPacket& packet = board->mSeedBank->mSeedPackets[s];
            SeedType seedType = packet.mPacketType;
            bool canAfford = false;
            if (packet.CanPickUp()) {
                int cost = board->GetCurrentPlantCost(seedType, SeedType::SEED_NONE);
                canAfford = board->CanTakeSunMoney(cost);
            }
            if (canAfford) {
                for (int row = 0; row < GRID_ROWS; row++) {
                    for (int col = 0; col < GRID_COLS; col++) {
                        if (board->CanPlantAt(col, row, seedType) == PlantingReason::PLANTING_OK) {
                            buf(PLANT_OFF + s * (GRID_ROWS * GRID_COLS) + row * GRID_COLS + col) = true;
                        }
                    }
                }
            }
        }

        return mask;
    }

    py::dict debug_sleep_state() const {
        py::dict out;
        Board* board = gLawnApp ? gLawnApp->mBoard : nullptr;
        if (!board) return out;
        out["background"] = static_cast<int>(board->mBackground);
        out["stage_is_night"] = board->StageIsNight();
        py::list plants;
        Plant* p = nullptr;
        while (board->mPlants.IterateNext(p)) {
            py::dict d;
            d["row"] = p->mRow;
            d["col"] = p->mPlantCol;
            d["seed"] = static_cast<int>(p->mSeedType);
            d["asleep"] = p->mIsAsleep;
            d["wake_counter"] = p->mWakeUpCounter;
            d["board_ok"] = p->mBoard != nullptr;
            if (p->mBoard) d["plant_board_night"] = p->mBoard->StageIsNight();
            plants.append(d);
        }
        out["plants"] = plants;
        return out;
    }

    py::dict get_render_state() const {
        py::dict state;
        state["scene"] = static_cast<int>(gLawnApp->mGameScene);
        state["has_board"] = gLawnApp->mBoard != nullptr;
        state["has_title_screen"] = gLawnApp->mTitleScreen != nullptr;
        state["loaded"] = gLawnApp->mLoaded;
        return state;
    }

    void setSunMoney(int sun) {
        if (gLawnApp && gLawnApp->mBoard) {
            gLawnApp->mBoard->mSunMoney = sun;
        }
    }

    void setSunPenaltyScale(float scale) {
        mSunPenaltyScale = scale;
    }

    void setCooldownScale(float scale) {
        gCooldownScale = scale;
    }

    py::dict get_obs() {
        const int channels = (mObsVersion >= 2) ? SPATIAL_CHANNELS + 2 : SPATIAL_CHANNELS;
        const int nGlobals = (mObsVersion >= 2) ? 26 : 24;
        auto spatial = py::array_t<float>(std::vector<py::ssize_t>({GRID_ROWS, GRID_COLS, channels}));
        auto global = py::array_t<float>({nGlobals});

        std::memset(spatial.mutable_data(), 0, spatial.nbytes());
        std::memset(global.mutable_data(), 0, global.nbytes());

        auto spatial_buf = spatial.mutable_unchecked<3>();
        auto global_buf = global.mutable_unchecked<1>();

        if (gLawnApp && gLawnApp->mBoard) {
            Board* board = gLawnApp->mBoard;

            // Plant data: channels 0-2
            Plant* p = nullptr;
            while (board->mPlants.IterateNext(p)) {
                if (p->mRow >= 0 && p->mRow < GRID_ROWS && p->mPlantCol >= 0 && p->mPlantCol < GRID_COLS) {
                    spatial_buf(p->mRow, p->mPlantCol, 0) = (float)p->mSeedType + 1.0f;
                    spatial_buf(p->mRow, p->mPlantCol, 1) = (float)p->mPlantHealth / (float)p->mPlantMaxHealth;
                    spatial_buf(p->mRow, p->mPlantCol, 2) = (float)p->mState;
                }
            }

            // Zombie data: channels 3..35 (per zombie type HP)
            Zombie* z = nullptr;
            while (board->mZombies.IterateNext(z)) {
                if (z->mRow >= 0 && z->mRow < GRID_ROWS && z->mZombieType >= 0 && z->mZombieType < NZOMBIE_TYPES) {
                    int col = (int)((z->mPosX - 40.0f) / 80.0f);
                    if (col < 0) col = 0;
                    if (col > 8) col = 8;
                    const int channel = PLANT_CHANNELS + (int)z->mZombieType;
                    assert(channel >= 0 && channel < SPATIAL_CHANNELS);
                    spatial_buf(z->mRow, col, channel) += (float)(z->mBodyHealth + z->mHelmHealth + z->mShieldHealth);
                }
            }

            // obs v2: normal-plant layer (channels 36-37). Deterministic
            // per-cell layering from GetPlantsOnLawn — reveals the plant
            // hidden under a Pumpkin shell.
            if (mObsVersion >= 2) {
                for (int row = 0; row < GRID_ROWS; row++) {
                    for (int col = 0; col < GRID_COLS; col++) {
                        PlantsOnLawn aPlantsOnLawn;
                        board->GetPlantsOnLawn(col, row, &aPlantsOnLawn);
                        Plant* normal = aPlantsOnLawn.mNormalPlant;
                        if (normal && !normal->mDead) {
                            spatial_buf(row, col, SPATIAL_CHANNELS) = (float)normal->mSeedType + 1.0f;
                            spatial_buf(row, col, SPATIAL_CHANNELS + 1) =
                                (float)normal->mPlantHealth / (float)normal->mPlantMaxHealth;
                        }
                    }
                }
            }

            // Global data
            global_buf(0) = (float)board->mSunMoney / 9990.0f;
            global_buf(1) = (float)board->mCurrentWave / (float)board->mNumWaves;

            if (board->mSeedBank) {
                for (int i = 0; i < 10; i++) {
                    if (i < board->mSeedBank->mNumPackets) {
                        auto& packet = board->mSeedBank->mSeedPackets[i];
                        // 0.0 = ready (mRefreshCounter is 0 both when ready
                        // and right after planting, so the previous
                        // "1 - counter/time" reported ready as 1.0 — same as
                        // just-planted — and made closer-to-ready read
                        // lower). Rising monotonically to 1.0 as the packet
                        // becomes ready again.
                        if (packet.mRefreshing && packet.mRefreshTime > 0)
                            global_buf(2 + i) = (float)packet.mRefreshCounter / (float)packet.mRefreshTime;
                        else
                            global_buf(2 + i) = 0.0f;
                    } else {
                        global_buf(2 + i) = 0.0f;
                    }
                }
            }

            // Markov-state details lost by the coarse 80px zombie grid.
            LawnMower* mower = nullptr;
            while (board->mLawnMowers.IterateNext(mower)) {
                if (mower->mRow >= 0 && mower->mRow < GRID_ROWS &&
                    !mower->mDead && mower->mMowerState == LawnMowerState::MOWER_READY) {
                    global_buf(12 + mower->mRow) = 1.0f;
                }
            }

            bool zombieInLane[GRID_ROWS] = {false, false, false, false, false};
            for (int row = 0; row < GRID_ROWS; row++) global_buf(17 + row) = 1.0f;
            z = nullptr;
            while (board->mZombies.IterateNext(z)) {
                float hp = (float)(z->mBodyHealth + z->mHelmHealth + z->mShieldHealth);
                if (hp > 0.0f && z->mRow >= 0 && z->mRow < GRID_ROWS) {
                    float x = std::clamp(z->mPosX / 900.0f, 0.0f, 1.0f);
                    if (!zombieInLane[z->mRow] || x < global_buf(17 + z->mRow)) {
                        global_buf(17 + z->mRow) = x;
                        zombieInLane[z->mRow] = true;
                    }
                }
            }
            global_buf(22) = std::clamp((float)board->mSunCountDown / 2500.0f, 0.0f, 1.0f);
            global_buf(23) = board->mZombieCountDownStart > 0
                ? std::clamp((float)board->mZombieCountDown / (float)board->mZombieCountDownStart, 0.0f, 1.0f)
                : 0.0f;
            if (mObsVersion >= 2) {
                // Absolute wave and stage: under chain_stages global[1]
                // resets at every stage boundary while zombie strength
                // scales with the stage index, so v2 exposes both.
                int wavesPerStage = board->GetNumWavesPerSurvivalStage();
                int absoluteWave =
                    board->mChallenge->mSurvivalStage * wavesPerStage + board->mCurrentWave + 1;
                global_buf(24) = (float)absoluteWave / (float)(wavesPerStage + 1);
                global_buf(25) = (float)board->mChallenge->mSurvivalStage;
            }
        }

        py::dict obs;
        obs["spatial"] = spatial;
        obs["global"] = global;
        return obs;
    }
};

PYBIND11_MODULE(pvz_env, m) {
    py::class_<PvZEnv>(m, "PvZEnv")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("resdir") = "pvz-portable/",
             py::arg("savedir") = "pvz-portable/savedata/")
        .def("reset", &PvZEnv::reset, py::arg("wave"), py::arg("seed") = -1)
        .def("step", &PvZEnv::step)
        .def("record_step", &PvZEnv::record_step, py::arg("action"), py::arg("frame_interval") = 5)
        .def("telemetry_step", &PvZEnv::telemetry_step, py::arg("action"), py::arg("frame_interval") = 5)
        .def("render", &PvZEnv::captureFrame)
        .def("get_render_state", &PvZEnv::get_render_state)
        .def("debug_sleep_state", &PvZEnv::debug_sleep_state)
        .def("get_obs", &PvZEnv::get_obs)
        .def("get_action_mask", &PvZEnv::get_action_mask)
        .def("squash_targetable_lanes", &PvZEnv::squashTargetableLanes)
        .def("set_plant_bonus_scale", &PvZEnv::set_plant_bonus_scale)
        .def("set_sun_money", &PvZEnv::setSunMoney)
        .def("set_sun_penalty_scale", &PvZEnv::setSunPenaltyScale)
        .def("set_cooldown_scale", &PvZEnv::setCooldownScale)
        .def("set_chain_stages", &PvZEnv::setChainStages)
        .def("set_obs_version", &PvZEnv::setObsVersion)
        .def("set_deck", &PvZEnv::setDeck)
        .def("debug_trigger_stage_end", &PvZEnv::debugTriggerStageEnd)
        .def_property_readonly_static("action_space_size", [](py::object) { return NUM_ACTIONS; });
}
